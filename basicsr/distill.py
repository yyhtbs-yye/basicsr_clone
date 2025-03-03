import datetime
import logging
import math
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

from basicsr.train import init_tb_loggers, create_train_val_dataloader, load_resume_state

from copy import deepcopy
from basicsr.utils.registry import MODEL_REGISTRY

class FeatureExtractor:
    def __init__(self, name):
        self.features = {}
        self.name = name

    def hook(self, module, input, output):

        # if self.name == 'student':
        #     print("module.layer_name: ", module.layer_name)
        # Save the output under a key determined by module.layer_name.
        self.features[module.layer_name] = output

    def clear(self):
        self.features.clear()

def build_teacher_model(opt):
    """Build teacher model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['teacher_type'])(opt)
    logger = get_root_logger()
    logger.info(f'Teacher Model [{model.__class__.__name__}] is created.')
    return model


def distill_pipeline(root_path):

    teacher_feature_extractor = None
    student_feature_extractor = None

    # Parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # Load resume states if necessary
    resume_state = load_resume_state(opt)

    # Create directories for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # Copy the YAML file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # Initialize the logger
    log_file = osp.join(opt['path']['log'], f"distill_{opt['name']}_{get_time_str()}.log")



    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # Initialize WandB and TensorBoard loggers
    tb_logger = init_tb_loggers(opt)

    # Create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # -------------------------------------------------------------------------
    #  Build the teacher model (if provided in the config).
    #  We will freeze it and use it for distillation guidance.
    # -------------------------------------------------------------------------

    logger.info(f'Building teacher model: {opt["teacher_type"]}')
    teacher_model = build_teacher_model(opt)

    logger.info(f'Building student model: {opt["model_type"]}')
    student_model = build_model(opt)

    # If resuming training, load resume state
    if resume_state:
        student_model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # Create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # Dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(
            f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'."
        )

    # ------------------
    #  Distillation Loop
    # ------------------

    def register_hooks(net, layers_to_extract, feature_extractor, tst):
        """Register forward hooks to capture features from the specified layers."""
        for name, module in net.named_modules():
            if name in layers_to_extract:
                module.layer_name = name  # so the hook knows which layer this is
                module.register_forward_hook(feature_extractor.hook)

    # Initialize feature extractors for teacher and student models
    teacher_feature_extractor = FeatureExtractor('teacher')
    student_feature_extractor = FeatureExtractor('student')

    # Register hooks for teacher and student models
    register_hooks(teacher_model.net_g_ema, opt['distill']['teacher_layers_to_extract'], teacher_feature_extractor, 'teacher')
    register_hooks(student_model.net_g, opt['distill']['student_layers_to_extract'], student_feature_extractor, 'student')

    pairing = {it:jt for it, jt in zip(opt['distill']['teacher_layers_to_extract'],
                                       opt['distill']['student_layers_to_extract'])}

    logger.info(f'Start distillation from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()
            current_iter += 1
            if current_iter > total_iters:
                break

            # Update learning rate
            student_model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1)
            )

            # ------------------------------------------------------
            # 1) Teacher forward (if teacher is available):
            #    We'll assume you want the teacher's output
            #    (e.g., a prediction) to guide the student.
            # ------------------------------------------------------

            if teacher_model is None:
                # quite with error
                raise ValueError("Teacher model is not provided, but student model requires it.")

            # Reset feature extractors
            teacher_feature_extractor.features = {}
            student_feature_extractor.features = {}

            # Forward teacher model
            teacher_model.feed_data(train_data)
            teacher_model.test()
            teacher_outputs = teacher_model.get_current_result()
            teacher_outputs.update(teacher_feature_extractor.features)

            student_model.feed_data(train_data)
            student_model.distill_parameters(current_iter, teacher_outputs, student_feature_extractor, pairing)

            # student_model.optimize_parameters(current_iter)

            iter_timer.record()

            if current_iter == 1:
                # Reset start time in msg_logger for more accurate eta_time
                # (Not working in resume mode)
                msg_logger.reset_start_time()

            # Logging
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': student_model.get_current_learning_rate()})
                log_vars.update({
                    'time': iter_timer.get_avg_time(),
                    'data_time': data_timer.get_avg_time()
                })
                log_vars.update(student_model.get_current_log())
                msg_logger(log_vars)

            # Save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                student_model.save(epoch, current_iter)

            # Validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning(
                        'Multiple validation datasets are *only* supported by SRModel.'
                    )
                for val_loader in val_loaders:
                    student_model.validation(
                        val_loader, current_iter, tb_logger, opt['val']['save_img']
                    )
                    teacher_model.validation(
                        val_loader, current_iter, tb_logger, opt['val']['save_img']
                    )

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter
    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training/distillation. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    student_model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest checkpoint

    # Final validation
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            student_model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    distill_pipeline(root_path)
