import functools
import torch
from torch.nn import functional as F

def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce the loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor, optional): A manual rescaling weight applied to the loss.
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'. Default: 'mean'.

    Returns:
        Tensor: Reduced loss.
    """
    if weight is not None:
        loss = loss * weight

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss

def get_gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for smoothing.

    Args:
        kernel_size (int): Size of the kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian.
        channels (int): Number of channels. The same kernel will be applied per channel.

    Returns:
        Tensor: A tensor of shape (channels, 1, kernel_size, kernel_size)
                containing the Gaussian kernel for each channel.
    """
    # Create a 1D tensor centered at 0.
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.
    # Create a 2D grid of (x,y) coordinates.
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    # Calculate the 2D Gaussian kernel.
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    # Reshape and repeat for each channel.
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel

def adaptive_loss(loss_func):
    """
    Decorator to create an adaptive loss function. The decorated loss function will:

    1. Compute the element-wise loss via the original loss function.
    2. Smooth the loss to obtain a focus map (via a Gaussian filter).
    3. Multiply the focus map with the loss so that areas with higher error contribute more.
    4. Apply optional weighting and reduction.

    The decorated function has the signature:

        loss_func(pred, target, weight=None, reduction='mean', kernel_size=3, sigma=1.0, **kwargs)

    Example usage:

        @adaptive_loss
        def mse_loss(pred, target):
            return F.mse_loss(pred, target, reduction='none')
    """
    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', kernel_size=3, sigma=1.0, **kwargs):
        # Compute the element-wise loss.
        loss = loss_func(pred, target, **kwargs)

        # If the loss is not 4D (e.g., [N, H, W]), add a channel dimension.
        if loss.dim() < 4:
            loss = loss.unsqueeze(1)  # Now shape becomes (N, 1, H, W)

        # Determine the number of channels.
        channels = loss.size(1)

        # Create the Gaussian kernel for smoothing.
        kernel = get_gaussian_kernel(kernel_size, sigma, channels).to(loss.device)

        # Smooth the loss to generate the focus map.
        focus_map = F.conv2d(loss, weight=kernel, groups=channels, padding=kernel_size // 2)

        # Multiply the original loss with the focus map to emphasize high-error regions.
        loss = loss * focus_map

        # If the original loss was not 4D, remove the added channel dimension.
        if loss.size(1) == 1 and pred.dim() < 4:
            loss = loss.squeeze(1)

        # Apply weighting and reduction.
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper
