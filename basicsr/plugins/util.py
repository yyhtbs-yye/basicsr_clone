import torch
import torch.nn as nn
from copy import deepcopy

def replace_modules(model, plugin_class, module_paths, config, frozen):

    def get_parent_module(model, module_name):
        """Return the parent module and attribute name for the given module name."""
        names = module_name.split('.')
        parent = model
        for name in names[:-1]:
            parent = getattr(parent, name)
        return parent, names[-1]

    for name, module in model.named_modules():
        if name in module_paths:
            # get the path for each plugin
            load_path = module_paths[name]

            # if it is already plugin class, ignore this module
            if not isinstance(module, plugin_class):
                # otherwise, get the module's parent
                parent, attr_name = get_parent_module(model, name)
                # Replace the module with the plugin-wrapped modul
                setattr(parent, attr_name, plugin_class(module, **config).cuda())

            # the plugin should be pretrained model
            load_network(module, load_path)

            # the plugin should not be trained (unless using adaptor, not considered here)
            for param in module.parameters():
                param.requires_grad = frozen

    return model

def load_network(module, load_path, param_key='params'):

    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        load_net = load_net[param_key]
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    module.load_state_dict(load_net, strict=False)