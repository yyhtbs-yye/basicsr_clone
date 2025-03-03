import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from basicsr.archs.linear_attn_util import *
from functools import partial

from basicsr.archs.rope_util import apply_rotary_emb, compute_axial_cis

class VanillaAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., base=10000):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.base = base

    def forward(self, x, img_size):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        out = self.proj(x)
        return out

class KernelAttention(nn.Module):

    def __init__(self, dim, num_heads, kernel_fn,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rope_theta=10.0, base=10000):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_fn = kernel_fn
        self.scale = qk_scale or dim**-0.5
        self.base = base

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.compute_cis = partial(compute_axial_cis, dim=self.head_dim, theta=rope_theta)

        # Initialize empty placeholders so we can store them
        self.freqs_cis_x = None
        self.freqs_cis_y = None


    def forward(self, x, img_size):
        b_, n, c = x.shape
        height, width = img_size

        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # --(A) Update RoPE cache if needed--
        if (self.freqs_cis_x is None) or (self.freqs_cis_x.shape[0] != n):
            # Recompute
            freqs_x, freqs_y = self.compute_cis(end_x=width, end_y=height)
            # shape: (height*width, D/2) in complex
            self.freqs_cis_x = freqs_x.to(x.device)
            self.freqs_cis_y = freqs_y.to(x.device)

        # --(B) Apply 2D axial RoPE: split q,k into x-part and y-part--

        # Split each of q_spatial, k_spatial along last dim
        half = self.head_dim // 2
        q_x, q_y = q[..., :half], q[..., half:]
        k_x, k_y = k[..., :half], k[..., half:]

        # Multiply each half by its corresponding freq
        q_x = apply_rotary_emb(q_x, self.freqs_cis_x)  # shape [B,H,N-1,D/2]
        q_y = apply_rotary_emb(q_y, self.freqs_cis_y)  # shape [B,H,N-1,D/2]
        k_x = apply_rotary_emb(k_x, self.freqs_cis_x)
        k_y = apply_rotary_emb(k_y, self.freqs_cis_y)

        # Re-concatenate the two halves => shape [B,H,N,D]
        q = torch.cat([q_x, q_y], dim=-1)
        k = torch.cat([k_x, k_y], dim=-1)

        # --(C) Kernel attention--
        # 1) Apply kernel function to queries and keys
        q = self.kernel_fn(q)
        k = self.kernel_fn(k)

        # 2) Combine k and v => shape [B, H, D, head_dim]
        kv = torch.einsum('b h n d, b h n e -> b h d e', k, v)  # (b, num_heads, head_dim, head_dim)

        # 3) For each token i, multiply q_i by (K'V)
        out = torch.einsum('b h n d, b h d e -> b h n e', q, kv)  # (b, num_heads, n, head_dim)

        # 4) Denominator => sum of k over n, then dot with q
        k_sum = k.sum(dim=2)  # (b, num_heads, head_dim)
        z = torch.einsum('b h n d, b h d -> b h n', q, k_sum)  # (b, num_heads, n)
        z = 1.0 / (z + 1e-6)  # Avoid division by zero

        # 5) Normalize
        out = out * z.unsqueeze(-1)  # (b, num_heads, n, head_dim)

        # 6) Merge heads => (B,N,C)
        out = out.transpose(1, 2).reshape(b_, n, self.dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


if __name__=="__main__":
    # Create instances of each module
    dim, num_heads = 512, 8
    input_tensor = torch.randn(1, 128*128, dim)

    # Compute FLOPs
    original_attention = VanillaAttention(dim, num_heads)
    # Compute FLOPs and parameters
    original_attention(input_tensor)
    flops, params = profile(original_attention, inputs=(input_tensor,))

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")

    # ELU is based on "Transformer dissection: An unified understanding for transformer’s attention via the lens of kernel"
    linear_attention = KernelAttention(dim, num_heads, kernel_fn=F.elu)
    # Compute FLOPs and parameters
    linear_attention(input_tensor)
    flops, params = profile(linear_attention, inputs=(input_tensor,))

    print(f"FLOPs: {flops}")
    print(f"Params: {params}")