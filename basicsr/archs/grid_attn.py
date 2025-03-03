import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5  # scaling factor for QK^T

    def forward(self, x):
        B_times_T, _ = x.size()

        # Project to Q, K, V
        q = self.q_proj(x).view(B_times_T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B_times_T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B_times_T, self.num_heads, self.head_dim)

        qk = torch.einsum('bhd,bHd->bhH', q, k) * self.scale  # shape: (B_times_T, num_heads, num_heads)
        attn = F.softmax(qk, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhH,bHd->bhd', attn, v)  # shape: (B_times_T, num_heads, head_dim)
        out = out.reshape(B_times_T, self.embed_dim)

        # Final projection
        out = self.out_proj(out)
        return out

class GridAttention(nn.Module):
    """
    Splits the feature map into non-overlapping patches (grid),
    applies self-attention within each patch, and folds them back.
    Ignores positional embeddings for simplicity.
    """
    def __init__(self, in_channels, patch_size=4, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.attn = SimpleMultiheadSelfAttention(
            embed_dim=in_channels,
            num_heads=num_heads
        )

        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)

        # We do not define the unfold/fold here because we need the spatial
        # dimensions at runtime.

    def forward(self, x): # x: (B, C, H, W) and Returns: (B, C, H, W)

        B, C, H, W = x.size()
        # 1) Unfold to get patches: shape => (B, C*patch_size*patch_size, num_patches)
        patches = self.unfold(x)  # (B, C*K*K, nP)
        # nP = (H/patch_size)*(W/patch_size)
        # K*K = patch_area

        # 2) Rearrange for per‐patch attention. We want each patch to be a "group" of tokens.
        patches = patches.permute(0, 2, 1)  # (B, nP, C*K*K)

        # Now we can treat each patch as (K*K) tokens, each dimension C, by reshaping:
        # The dimension is nP * B for the "batch," and K*K is the "time."
        # So let's do:
        patches = patches.reshape(B * patches.shape[1], C, self.patch_size*self.patch_size)
        # shape => (B*nP, C, K*K)

        # Then we transpose to (B*nP, K*K, C) so each row is a token vector of size C
        patches = patches.transpose(1, 2)  # (B*nP, K*K, C)

        # 3) Apply self-attention within each patch. We'll flatten the time dimension as well.
        # We want to let each patch's (K*K) tokens attend among themselves.
        # We'll do that by flattening or by iterating. For minimal code, let's flatten:
        BnP, Tk, C_ = patches.shape
        # We'll treat BnP as the "batch of patches," and we want to pass (BnP*Tk, C_) to attention
        patches = patches.reshape(BnP*Tk, C_)
        # Now do attention
        out = self.attn(patches)
        # shape => (BnP*Tk, C_)

        # 4) Reshape back
        out = out.reshape(BnP, Tk, C_).transpose(1, 2)  # (BnP, C_, Tk) => (BnP, C_, K*K)

        # 5) Fold patches back into (B, C, H, W)
        out = out.reshape(B, -1, self.patch_size*self.patch_size, C_)
        # Actually, it's easier to invert step 2 carefully. We will do a direct fold operation:
        out = out.reshape(B*nP, C_, self.patch_size*self.patch_size)
        out = out.transpose(1, 2)  # (B*nP, patch_size*patch_size, C_)
        # Flatten the channel dimension for folding:
        out = out.reshape(B, patches.shape[0]//B, C*self.patch_size*self.patch_size)
        out = out.permute(0, 2, 1)

        # Use nn.Fold to reverse the operation of Unfold
        fold = nn.Fold(output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)
        out = fold(out)

        return out

def grid_shuffle(x, h, w, c, interval_size):
    x = x.view(-1, h // interval_size, interval_size, w // interval_size, interval_size, c)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.view(-1, h // interval_size, w // interval_size, c)

    return x

def grid_unshuffle(x, b, h, w, interval_size):
    x = x.view(b, interval_size, interval_size, h // interval_size, w // interval_size, -1)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, h, w, -1)
    return x
