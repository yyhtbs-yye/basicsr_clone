# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel

from basicsr.utils.misc import set_random_seed
from PIL import Image
from tqdm.auto import tqdm
from basicsr.archs import build_network
from basicsr.diffusers.schedulers import build_scheduler
from basicsr.tokenizer.tokenizer_wrapper import CLIPTokenizer

from basicsr.diffusers.utils.stable_diffusion_utils import set_tomesd, set_xformers

ModelType = Union[Dict, nn.Module]

@MODEL_REGISTRY.register()
class StableDiffusion(BaseModel):
    """Class for Stable Diffusion. Refers to https://github.com/Stability-
    AI/stablediffusion and https://github.com/huggingface/diffusers/blob/main/s
    rc/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_attend_an
    d_excite.py  # noqa.

    Args: opt
    """

    def __init__(self, opt):

        # TODO: support `from_pretrained` for this class
        super().__init__(opt) # There might be some preprocessor here!!!

        default_args = dict()
        if opt['dtype'] is not None:
            default_args['dtype'] = opt['dtype']

        self.dtype = torch.float32
        if opt['dtype'] in ['float16', 'fp16', 'half']:
            self.dtype = torch.float16
        elif opt['dtype'] == 'bf16':
            self.dtype = torch.bfloat16
        else:
            assert opt['dtype'] in [
                'fp32', None
            ], ('dtype must be one of \'fp32\', \'fp16\', \'bf16\' or None.')

        self.vae = build_network(opt['network_vae'])
        self.unet = build_network(opt['network_unet']) # NOTE: initialize unet as fp32
        self.text_encoder = build_network(opt['text_encoder'])

        self._unet_ori_dtype = next(self.unet.parameters()).dtype

        print(f'Set UNet dtype to \'{self._unet_ori_dtype}\'.', 'current')

        self.scheduler = build_scheduler(opt['diffusion_scheduler'])

        if opt['test_scheduler'] is None:
            self.test_scheduler = deepcopy(self.scheduler)
        else:
            self.test_scheduler = build_scheduler(opt['test_scheduler'])

        # NOTE: here we assume tokenizer is an string
        self.tokenizer = CLIPTokenizer(opt['tokenizer'], subfolder='tokenizer')

        self.unet_sample_size = self.unet.sample_size
        self.vae_scale_factor = 2**(len(self.vae.block_out_channels) - 1)

        self.enable_noise_offset = opt['noise_offset_weight'] > 0
        self.noise_offset_weight = opt['noise_offset_weight']

        self.enable_xformers = opt['enable_xformers']
        self.set_xformers()

        self.tomesd_cfg = opt['tomesd_cfg']
        self.set_tomesd()

    def set_xformers(self, module: Optional[nn.Module] = None) -> nn.Module:
        """Set xformers for the model.

        Returns:
            nn.Module: The model with xformers.
        """
        if self.enable_xformers:
            if module is None:
                set_xformers(self)
            else:
                set_xformers(module)

    def set_tomesd(self) -> nn.Module:
        """Set ToMe for the stable diffusion model.

        Returns:
            nn.Module: The model with ToMe.
        """
        if self.tomesd_cfg is not None:
            set_tomesd(self, **self.tomesd_cfg)

    @property
    def device(self):
        return next(self.parameters()).device

    def train(self, mode: bool = True):
        """Set train/eval mode.

        Args:
            mode (bool, optional): Whether set train mode. Defaults to True.
        """
        if mode:
            if next(self.unet.parameters()).dtype != self._unet_ori_dtype:
                print(
                    f'Set UNet dtype to \'{self._unet_ori_dtype}\' '
                    'in the train mode.', 'current')
            self.unet.to(self._unet_ori_dtype)
        else:
            self.unet.to(self.dtype)
            print(f'Set UNet dtype to \'{self.dtype}\' in the eval mode.',
                      'current')
        return super().train(mode)

    @torch.no_grad()
    def infer(self,
              prompt: Union[str, List[str]],
              height: Optional[int] = None,
              width: Optional[int] = None,
              num_inference_steps: int = 50,
              guidance_scale: float = 7.5,
              negative_prompt: Optional[Union[str, List[str]]] = None,
              num_images_per_prompt: Optional[int] = 1,
              eta: float = 0.0,
              generator: Optional[torch.Generator] = None,
              latents: Optional[torch.FloatTensor] = None,
              show_progress=True,
              seed=1,
              return_type='image'):

        assert return_type in ['image', 'tensor', 'numpy']
        set_random_seed(seed=seed)

        # 0. Default height and width to unet
        height = height or self.unet_sample_size * self.vae_scale_factor
        width = width or self.unet_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self.device

        img_dtype = self.vae.module.dtype if hasattr(self.vae, 'module') \
            else self.vae.dtype
        latent_dtype = next(self.unet.parameters()).dtype
        # here `guidance_scale` is defined analog to the
        # guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf .
        # `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, device,
                                              num_images_per_prompt,
                                              do_classifier_free_guidance,
                                              negative_prompt)

        # 4. Prepare timesteps
        self.test_scheduler.set_timesteps(num_inference_steps)
        timesteps = self.test_scheduler.timesteps

        # 5. Prepare latent variables
        if hasattr(self.unet, 'module'):
            num_channels_latents = self.unet.module.in_channels
        else:
            num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        # TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        if show_progress:
            timesteps = tqdm(timesteps)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.test_scheduler.scale_model_input(
                latent_model_input, t)

            latent_model_input = latent_model_input.to(latent_dtype)
            text_embeddings = text_embeddings.to(latent_dtype)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t,
                encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.test_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs)['prev_sample']

        # 8. Post-processing
        image = self.decode_latents(latents.to(img_dtype))
        if return_type == 'image':
            image = self.output_to_pil(image)
        elif return_type == 'numpy':
            image = image.cpu().numpy()
        else:
            assert return_type == 'tensor', (
                'Only support \'image\', \'numpy\' and \'tensor\' for '
                f'return_type, but receive {return_type}')

        return {'samples': image}

    def output_to_pil(self, image) -> List[Image.Image]:
        """Convert output tensor to PIL image. Output tensor will be de-normed
        to [0, 255] by `DataPreprocessor.destruct`. Due to no `data_samples` is
        passed, color order conversion will not be performed.

        Args:
            image (torch.Tensor): The output tensor of the decoder.

        Returns:
            List[Image.Image]: The list of processed PIL images.
        """
        image = self.data_preprocessor.destruct(image)
        image = image.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        image = [Image.fromarray(img) for img in image]
        return image

    def _encode_prompt(self, prompt, device, num_images_per_prompt,
                       do_classifier_free_guidance, negative_prompt):
        """Encodes the prompt into text encoder hidden states.

        Args:
            prompt (str or list(int)): prompt to be encoded.
            device: (torch.device): torch device.
            num_images_per_prompt (int): number of images that should be
                generated per prompt.
            do_classifier_free_guidance (`bool`): whether to use classifier
                free guidance or not.
            negative_prompt (str or List[str]): The prompt or prompts not
                to guide the image generation. Ignored when not using
                guidance (i.e., ignored if `guidance_scale` is less than `1`).

        Returns:
            text_embeddings (torch.Tensor): text embeddings generated by
                clip text encoder.
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding='max_length', return_tensors='pt').input_ids

        if not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1:-1])
            print(
                'The following part of your input was truncated because CLIP'
                ' can only handle sequences up to'
                f' {self.tokenizer.model_max_length} tokens: {removed_text}')

        text_encoder = self.text_encoder.module if hasattr(
            self.text_encoder, 'module') else self.text_encoder
        if hasattr(text_encoder.config, 'use_attention_mask'
                   ) and text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt,
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [''] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f'`negative_prompt` should be the same type to `prompt`,'
                    f'but got {type(negative_prompt)} !='
                    f' {type(prompt)}.')
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f'`negative_prompt`: {negative_prompt} has '
                    f'batch size {len(negative_prompt)}, but `prompt`:'
                    f' {prompt} has batch size {batch_size}.'
                    f' Please make sure that passed `negative_prompt` matches'
                    ' the batch size of `prompt`.')
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding='max_length',
                max_length=max_length,
                truncation=True,
                return_tensors='pt',
            )

            if hasattr(text_encoder.config, 'use_attention_mask'
                       ) and text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for
            # each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(
                1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional
            # and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents):
        """use vae to decode latents.

        Args:
            latents (torch.Tensor): latents to decode.

        Returns:
            image (torch.Tensor): image result.
        """
        latents = 1 / 0.18215 * latents
        if hasattr(self.vae, 'module'):
            image = self.vae.module.decode(latents)['sample']
        else:
            image = self.vae.decode(latents)['sample']
        # we always cast to float32 as this does not cause
        # significant overhead and is compatible with bfloa16
        return image.float()

    def prepare_extra_step_kwargs(self, generator, eta):
        """prepare extra kwargs for the scheduler step.

        Args:
            generator (torch.Generator):
                generator for random functions.
            eta (float):
                eta (η) is only used with the DDIMScheduler,
                it will be ignored for other schedulers.
                eta corresponds to η in DDIM paper:
                https://arxiv.org/abs/2010.02502
                and should be between [0, 1]

        Return:
            extra_step_kwargs (dict):
                dict contains 'generator' and 'eta'
        """
        accepts_eta = 'eta' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        # check if the scheduler accepts generator
        accepts_generator = 'generator' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def prepare_test_scheduler_extra_step_kwargs(self, generator, eta):
        """prepare extra kwargs for the scheduler step.

        Args:
            generator (torch.Generator):
                generator for random functions.
            eta (float):
                eta (η) is only used with the DDIMScheduler,
                it will be ignored for other schedulers.
                eta corresponds to η in DDIM paper:
                https://arxiv.org/abs/2010.02502
                and should be between [0, 1]

        Return:
            extra_step_kwargs (dict):
                dict contains 'generator' and 'eta'
        """
        accepts_eta = 'eta' in set(
            inspect.signature(self.test_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        # check if the scheduler accepts generator
        accepts_generator = 'generator' in set(
            inspect.signature(self.test_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width):
        """check whether inputs are in suitable format or not."""

        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f'`prompt` has to be of '
                             f'type `str` or `list` but is {type(prompt)}')

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f'`height` and `width` have to be divisible '
                             f'by 8 but are {height} and {width}.')

    def prepare_latents(self,
                        batch_size,
                        num_channels_latents,
                        height,
                        width,
                        dtype,
                        device,
                        generator,
                        latents=None):
        """prepare latents for diffusion to run in latent space.

        Args:
            batch_size (int): batch size.
            num_channels_latents (int): latent channel nums.
            height (int): image height.
            width (int): image width.
            dtype (torch.dtype): float type.
            device (torch.device): torch device.
            generator (torch.Generator):
                generator for random functions, defaults to None.
            latents (torch.Tensor):
                Pre-generated noisy latents, defaults to None.

        Return:
            latents (torch.Tensor): prepared latents.
        """
        shape = (batch_size, num_channels_latents,
                 height // self.vae_scale_factor,
                 width // self.vae_scale_factor)
        if latents is None:
            latents = torch.randn(
                shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f'Unexpected latents shape, '
                                 f'got {latents.shape}, expected {shape}')
            latents = latents.to(device)

        # scale the initial noise by the standard
        # deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def val_step(self, data: dict):
        data = self.data_preprocessor(data)
        data_samples = data['data_samples']
        prompt = data_samples.prompt

        output = self.infer(prompt, return_type='tensor')
        samples = output['samples']

        samples = self.data_preprocessor.destruct(samples, data_samples)
        gt_img = self.data_preprocessor.destruct(data['inputs'], data_samples)

        out_data_sample = DataSample(
            fake_img=samples, gt_img=gt_img, prompt=prompt)

        # out_data_sample = DataSample(fake_img=samples, prompt=prompt)
        data_sample_list = out_data_sample.split()
        return data_sample_list

    @torch.no_grad()
    def test_step(self, data: dict):
        data = self.data_preprocessor(data)
        data_samples = data['data_samples']
        prompt = data_samples.prompt

        output = self.infer(prompt, return_type='tensor')
        samples = output['samples']

        samples = self.data_preprocessor.destruct(samples, data_samples)
        gt_img = self.data_preprocessor.destruct(data['inputs'], data_samples)

        out_data_sample = DataSample(
            fake_img=samples, gt_img=gt_img, prompt=prompt)
        data_sample_list = out_data_sample.split()
        return data_sample_list

    def train_step(self, data, optim_wrapper_dict):
        data = self.data_preprocessor(data)
        inputs, data_samples = data['inputs'], data['data_samples']

        vae = self.vae.module if hasattr(self.vae, 'module') else self.vae

        optim_wrapper = optim_wrapper_dict['unet']
        with optim_wrapper.optim_context(self.unet):
            image = inputs
            prompt = data_samples.prompt
            num_batches = image.shape[0]

            image = image.to(self.dtype)
            latents = vae.encode(image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)

            if self.enable_noise_offset:
                noise = noise + self.noise_offset_weight * torch.randn(
                    latents.shape[0], latents.shape[1],
                    1, 1,
                    device=noise.device)

            timesteps = torch.randint(
                0,
                self.scheduler.num_train_timesteps, (num_batches, ),
                device=self.device)
            timesteps = timesteps.long()

            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            input_ids = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                return_tensors='pt',
                padding='max_length',
                truncation=True)['input_ids'].to(self.device)

            encoder_hidden_states = self.text_encoder(input_ids)[0]

            if self.scheduler.config.prediction_type == 'epsilon':
                gt = noise
            elif self.scheduler.config.prediction_type == 'v_prediction':
                gt = self.scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError('Unknown prediction type '
                                 f'{self.scheduler.config.prediction_type}')

            # NOTE: we train unet in fp32, convert to float manually
            model_output = self.unet(
                noisy_latents.float(),
                timesteps,
                encoder_hidden_states=encoder_hidden_states.float())
            model_pred = model_output['sample']

            loss_dict = dict()
            # calculate loss in FP32
            loss_mse = F.mse_loss(model_pred.float(), gt.float())
            loss_dict['loss_mse'] = loss_mse

            parsed_loss, log_vars = self.parse_losses(loss_dict)
            optim_wrapper.update_params(parsed_loss)

        return log_vars

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        """forward is not implemented now."""
        raise NotImplementedError(
            'Forward is not implemented now, please use infer.')
