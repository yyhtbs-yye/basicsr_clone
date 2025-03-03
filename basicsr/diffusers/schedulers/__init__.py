import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import SCHEDULER_REGISTRY
import ddim_scheduler, ddpm_scheduler

__all__ = ['ddim_scheduler', 'ddpm_scheduler']

# automatically scan and import scheduler modules for registry
# scan all the files under the 'schedulers' folder and collect files ending with '_scheduler.py'
scheduler_folder = osp.dirname(osp.abspath(__file__))
scheduler_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(scheduler_folder) if v.endswith('_scheduler.py')]
# import all the scheduler modules
_model_modules = [importlib.import_module(f'basicsr.diffusers.schedulers.{file_name}') for file_name in scheduler_filenames]


def build_scheduler(opt):
    """Build scheduler from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    scheduler_type = opt.pop('type')
    scheduler = SCHEDULER_REGISTRY.get(scheduler_type)(**opt)
    logger = get_root_logger()
    logger.info(f'diffusion scheduler [{scheduler.__class__.__name__}] is created.')
    return scheduler
