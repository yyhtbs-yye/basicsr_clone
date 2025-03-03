import torch
import torch.nn as nn
from basicsr.diffusers.utils.tome_utils import build_mmagic_tomesd_block, build_mmagic_wrapper_tomesd_block

def isinstance_str(x: object, cls_name: str):
    """Checks whether `x` has any class *named* `cls_name` in its ancestry.
    Doesn't require access to the class's implementation.

    Source: https://github.com/dbolya/tomesd/blob/main/tomesd/utils.py#L3 # noqa
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False

def add_tome_cfg_hook(model: torch.nn.Module):
    """Add a forward pre hook to get the image size. This hook can be removed
    with remove_patch.

    Source: https://github.com/dbolya/tomesd/blob/main/tomesd/patch.py#L158 # noqa
    """

    def hook(module, args):
        module._tome_info['size'] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info['hooks'].append(model.register_forward_pre_hook(hook))

def set_xformers(module: nn.Module, prefix: str = '') -> nn.Module:
    """Set xformers' efficient Attention for attention modules.

    Args:
        module (nn.Module): The module to set xformers.
        prefix (str): The prefix of the module name.

    Returns:
        nn.Module: The module with xformers' efficient Attention.
    """

    if not xformers_is_enable():
        print('Do not support Xformers. Please install Xformers first. '
                  'The program will run without Xformers.')
        return

    for n, m in module.named_children():
        if hasattr(m, 'set_use_memory_efficient_attention_xformers'):
            # set xformers for Diffusers' Cross Attention
            m.set_use_memory_efficient_attention_xformers(True)
            module_name = f'{prefix}.{n}' if prefix else n
            print(
                'Enable Xformers for HuggingFace Diffusers\' '
                f'module \'{module_name}\'.', 'current')
        else:
            set_xformers(m, prefix=n)

    return module

def set_tomesd(model: torch.nn.Module,
               ratio: float = 0.5,
               max_downsample: int = 1,
               sx: int = 2,
               sy: int = 2,
               use_rand: bool = True,
               merge_attn: bool = True,
               merge_crossattn: bool = False,
               merge_mlp: bool = False):
    """Patches a stable diffusion model with ToMe. Apply this to the highest
    level stable diffusion object.

    Refer to: https://github.com/dbolya/tomesd/blob/main/tomesd/patch.py#L173 # noqa

    Args:
        model (torch.nn.Module): A top level Stable Diffusion module to patch in place.
        ratio (float): The ratio of tokens to merge. I.e., 0.4 would reduce the total
            number of tokens by 40%.The maximum value for this is 1-(1/(`sx` * `sy`)). By default,
            the max ratio is 0.75 (usually <= 0.5 is recommended). Higher values result in more speed-up,
            but with more visual quality loss.
        max_downsample (int): Apply ToMe to layers with at most this amount of downsampling.
            E.g., 1 only applies to layers with no downsampling, while 8 applies to all layers.
            Should be chosen from [1, 2, 4, or 8]. 1 and 2 are recommended.
        sx, sy (int, int): The stride for computing dst sets. A higher stride means you can merge
            more tokens, default setting of (2, 2) works well in most cases.
            `sx` and `sy` do not need to divide image size.
        use_rand (bool): Whether or not to allow random perturbations when computing dst sets.
            By default: True, but if you're having weird artifacts you can try turning this off.
        merge_attn (bool): Whether or not to merge tokens for attention (recommended).
        merge_crossattn (bool): Whether or not to merge tokens for cross attention (not recommended).
        merge_mlp (bool): Whether or not to merge tokens for the mlp layers (particular not recommended).

    Returns:
        model (torch.nn.Module): Model patched by ToMe.
    """

    # Make sure the module is not currently patched
    remove_tomesd(model)

    is_mmagic = isinstance_str(model, 'StableDiffusion') or isinstance_str(
        model, 'BaseModel')

    if is_mmagic:
        # Supports "StableDiffusion.unet" and "unet"
        diffusion_model = model.unet if hasattr(model, 'unet') else model
        if isinstance_str(diffusion_model, 'DenoisingUnet'):
            is_wrapper = False
        else:
            is_wrapper = True
    else:
        if not hasattr(model, 'model') or not hasattr(model.model,
                                                      'diffusion_model'):
            # Provided model not supported
            print('Expected a Stable Diffusion / Latent Diffusion model.')
            raise RuntimeError('Provided model was not supported.')
        diffusion_model = model.model.diffusion_model
        # TODO: can support more diffusion models, like Stability AI
        is_wrapper = None

    diffusion_model._tome_info = {
        'size': None,
        'hooks': [],
        'args': {
            'ratio': ratio,
            'max_downsample': max_downsample,
            'sx': sx,
            'sy': sy,
            'use_rand': use_rand,
            'merge_attn': merge_attn,
            'merge_crossattn': merge_crossattn,
            'merge_mlp': merge_mlp
        }
    }
    add_tome_cfg_hook(diffusion_model)

    for _, module in diffusion_model.named_modules():
        if isinstance_str(module, 'BasicTransformerBlock'):
            # TODO: can support more stable diffusion based models
            if is_mmagic:
                if is_wrapper is None:
                    raise NotImplementedError(
                        'Specific ToMe block not implemented')
                elif not is_wrapper:
                    make_tome_block_fn = build_mmagic_tomesd_block
                elif is_wrapper:
                    make_tome_block_fn = build_mmagic_wrapper_tomesd_block
            else:
                raise TypeError(
                    'Currently `tome` only support *stable-diffusion* model!')
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = diffusion_model._tome_info

    return model


def remove_tomesd(model: torch.nn.Module):
    """Removes a patch from a ToMe Diffusion module if it was already patched.

    Refer to: https://github.com/dbolya/tomesd/blob/main/tomesd/patch.py#L251 # noqa
    """
    # For mmagic Stable Diffusion models
    model = model.unet if hasattr(model, 'unet') else model

    for _, module in model.named_modules():
        if hasattr(module, '_tome_info'):
            for hook in module._tome_info['hooks']:
                hook.remove()
            module._tome_info['hooks'].clear()

        if module.__class__.__name__ == 'ToMeBlock':
            module.__class__ = module._parent

    return model