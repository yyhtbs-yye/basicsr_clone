import torch
import torch.nn as nn
from einops import rearrange

class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # Compute queries, keys, and values
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class InterPatchAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, attn_drop, proj_drop,
                 patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.msa = MultiheadSelfAttention(dim*patch_size*patch_size, num_heads=num_heads,
                                            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

    def forward(self, x, img_size):
        height, width = img_size
        x = patch_embed(x, height, width, self.patch_size)
        x = self.msa(x)
        x = patch_unembed(x, height, width, self.patch_size)

        return x

# [B, H, W, C] -> [B, (H//P) * (W//P), C*P*P]
def patch_embed(x, height, width, patch_size):
    n_h = height // patch_size
    n_w = width // patch_size
    x = rearrange(x, 'b (h ps1 w ps2) c -> b (h w) (ps1 ps2 c)',
                  ps1=patch_size, ps2=patch_size, h=n_h, w=n_w)

    return x

# [B, (H//P) * (W//P), C*P*P] -> [B, H, W, C]
def patch_unembed(x, height, width, patch_size):
    n_h = height // patch_size
    n_w = width // patch_size

    x = rearrange(x, 'b (h w) (ps1 ps2 c) -> b (h ps1 w ps2) c',
                  ps1=patch_size, ps2=patch_size, h=n_h, w=n_w)
    return x
