import torch
import torch.nn as nn
from copy import deepcopy
import os

def get_parent_module(model, module_name):
    """Return the parent module and attribute name for the given module name."""
    names = module_name.split('.')
    parent = model
    for name in names[:-1]:
        parent = getattr(parent, name)
    return parent, names[-1]


def replace_modules(model_a, model_b, module_mapping):

    device = next(model_a.parameters()).device

    for module_path_a, module_path_b in module_mapping.items():
        parent_a, attr_name_a = get_parent_module(model_a, module_path_a)
        module_b = model_b.get_submodule(module_path_b).to(device)
        setattr(parent_a, attr_name_a, module_b)

    return model_a

def load_modules(model, load_paths, param_key=None):

    for module_path, load_path in load_paths.items():

        module = model.get_submodule(module_path)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        # Adjust keys that start with 'module.' and build an adjusted dictionary.
        adjusted_load_net = {}
        for k, v in load_net.items():
            new_key = k[7:] if k.startswith('module.') else k
            adjusted_load_net[new_key] = v

        # Load the state dictionary into the module
        module.load_state_dict(adjusted_load_net, strict=True)

        # Retrieve the module's current state
        module_state = module.state_dict()

        # Verify that every parameter in adjusted_load_net matches the module state
        print("Verifying parameter equality:")
        all_match = True
        for key, loaded_value in adjusted_load_net.items():
            loaded_value = loaded_value.to(next(module.named_parameters())[1].device)
            module_value = module_state.get(key)
            if module_value is None:
                print(f"Parameter '{key}' not found in module state!")
                all_match = False
            elif not torch.allclose(module_value, loaded_value, rtol=1e-05, atol=1e-08):
                diff = (module_value - loaded_value).abs().sum().item()
                print(f"Parameter '{key}' does NOT match! Sum of absolute differences: {diff:.6f}")
                all_match = False
            else:
                print(f"Parameter '{key}' matches.")

        if all_match:
            print(f"All parameters match for module '{module_path}'.")
        else:
            print(f"Some parameters did not match for module '{module_path}'.")
        print("-" * 50)

    return model

def manage_freezing(model, freeze_module_paths, unfreeze_module_paths):

    for freeze_module_path in freeze_module_paths:
        for param in model.get_submodule(freeze_module_path).parameters():
            param.requires_grad = False

        print(f"Module '{freeze_module_path}' is frozen.")

    for unfreeze_module_path in unfreeze_module_paths:
        for param in model.get_submodule(unfreeze_module_path).parameters():
            param.requires_grad = True

        print(f"Module '{unfreeze_module_path}' is unfrozen.")

    return model

