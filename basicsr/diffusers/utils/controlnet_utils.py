# Copyright (c) OpenMMLab. All rights reserved.
from logging import ERROR
from typing import Optional

import torch.nn as nn
from basicsr.utils import get_root_logger # Maybe good

def change_base_model(controlnet: nn.Module,
                      curr_model: nn.Module,
                      base_model: nn.Module,
                      *args,
                      **kwargs) -> nn.Module:
    """This function is used to change the base model of ControlNet. Refers to
    https://github.com/lllyasviel/ControlNet/blob/main/tool_transfer_control.py
    .

    # noqa.

    Args:
        controlnet (nn.Module): The model for ControlNet to convert.
        curr_model (nn.Module): The model of current Stable Diffusion's Unet.
        base_model (nn.Module): The model of Stable Diffusion's Unet which
            ControlNet initialized with.
        save_path (str, optional): The path to save the converted model.
            Defaults to None.

        *args, **kwargs: Arguments for `save_checkpoint`.
    """
    dtype = next(controlnet.parameters()).dtype
    base_state_dict = base_model.state_dict()
    curr_state_dict = curr_model.state_dict()

    print('Start convert ControlNet to new Unet.', 'current')
    for k, v in controlnet.state_dict().items():
        if k in base_state_dict:
            base_v = base_state_dict[k].cpu()
            curr_v = curr_state_dict[k].cpu()
            try:
                offset = v.cpu() - base_v
                new_v = offset + curr_v
                controlnet.state_dict()[k].data.copy_(new_v.to(dtype))
                print(f'Convert success: \'{k}\'.', 'current')
            except Exception as exception:
                print(
                    f'Error occurs when convert \'{k}\'. '
                    'Please check that the model structure of '
                    '\'ControlNet\', \'BaseModel\' and \'CurrentModel\' '
                    'are consistent.', 'current', ERROR)
                raise exception

    return controlnet