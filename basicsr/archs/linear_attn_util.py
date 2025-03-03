
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

###############################################
# Rotary Positional Encoding Helpers
###############################################
def rotate_every_two(x):
    """
    Rotates every pair of elements in the last dimension.
    x: (..., head_dim) with head_dim assumed even.
    """
    # Split the last dimension into two halves
    x1 = x[..., ::2]  # even indices
    x2 = x[..., 1::2] # odd indices
    # Apply rotation: (x1, x2) -> (-x2, x1)
    x_rotated = torch.stack((-x2, x1), dim=-1)
    return x_rotated.flatten(-2)

def get_rotary_embedding(seq_len, dim, base=10000):
    """
    Computes rotary embeddings.
    Returns cos and sin tensors of shape (seq_len, dim)
    where dim should be the head dimension.
    """
    # dim is the head dimension (must be even)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len).float()
    sinusoid_inp = torch.einsum('i,j->ij', positions, inv_freq)  # (seq_len, dim/2)
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    # Expand to match head dimension by interleaving
    sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
    cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
    return cos, sin

def apply_rotary_pos_emb(x, cos, sin):
    """
    Applies rotary positional encoding.
    x: (batch, num_heads, seq_len, head_dim)
    cos, sin: (1, 1, seq_len, head_dim) or broadcastable to x.
    """
    return x * cos + rotate_every_two(x) * sin
