import torch
import torch.nn as nn
import torchvision.models as models

class Adaptor(nn.Module):
    def __init__(self, original_linear, rank, alpha=1.0):
        super().__init__()
        # Copy some metadata
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.bias = (original_linear.bias is not None)
        self.alpha = alpha

        # Store the base weight (frozen or not, depending on your needs)
        self.weight = original_linear.weight.detach()  # or clone it if you want
        self.bias_ = original_linear.bias.detach()     # rename to avoid name clash

        # Create LoRA parameters
        self.A = nn.Parameter(torch.zeros(self.out_features, rank), requires_grad=True)
        self.B = nn.Parameter(torch.zeros(rank, self.in_features), requires_grad=True)
        nn.init.normal_(self.A, std=0.02)
        nn.init.normal_(self.B, std=0.02)

    def forward(self, x):
        # W + alpha * (A @ B)
        effective_weight = self.weight + self.alpha * (self.A @ self.B)
        return nn.functional.linear(x, effective_weight, self.bias_)
