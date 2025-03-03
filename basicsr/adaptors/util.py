import torch
import torch.nn as nn

def apply_adaptor_to_layers(model, adaptor_class, layer_names, config):

    def get_parent_module(model, module_name):
        """Return the parent module and attribute name for the given module name."""
        names = module_name.split('.')
        parent = model
        for name in names[:-1]:
            parent = getattr(parent, name)
        return parent, names[-1]

    for name, module in model.named_modules():
        if name in layer_names:
            if not isinstance(module, adaptor_class):
                parent, attr_name = get_parent_module(model, name)
                # Replace the module with the adaptor-wrapped module
                setattr(parent, attr_name, adaptor_class(module, **config).cuda())
        else:
            for param in module.parameters():
                param.requires_grad = False

    return model


if __name__=="__main__":
    import torchvision.models as models
    from basicsr.adaptors.lora_linear import Adaptor

    # We'll pick resnet18 for simplicity
    model = models.resnet18()


    # Patch the final fully connected layer ("fc")
    model = apply_adaptor_to_layers(model, Adaptor, ["fc"], dict(rank=4, alpha=1.0))

    x = torch.randn(1, 3, 224, 224)
    output = model(x)  # triggers forward + forward_hook
    target = torch.tensor([1])  # Example target class index
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)

    model.zero_grad()

    # Backward pass
    loss.backward()

    # Check gradients
    for name, parameter in model.named_parameters():
        print(f"{name} gradient: \n {parameter.grad}")
