# --- In some file "swinir_router.py" ---
from basicsr.archs.swinir_arch import SwinIR  # Base class  :contentReference[oaicite:0]{index=0}
from basicsr.archs.linear_attn import KernelAttention, VanillaAttention
from basicsr.archs.patch_attn import InterPatchAttention
from basicsr.utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn
import torch.nn.functional as F

@ARCH_REGISTRY.register()
class GwinIR(SwinIR):

    def __init__(self, router_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # We'll create a Router module for each "stage" in self.layers
        self.num_stages = self.num_layers
        if router_args is None:
            router_args = {}

        self.router_routers = nn.ModuleList([
            VanillaAttention(dim=self.embed_dim, num_heads=self.num_heads[i],
                                         qkv_bias=True, attn_drop=0., proj_drop=0.)
            for i in range(self.num_layers)
        ])

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward_features(self, x):
        B, D, H, W = x.shape

        # Permute (B, D, H, W) -> (B, H*W, D)
        x = x.view(B, D, -1).permute(0, 2, 1)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer, router in zip(self.layers, self.router_routers):
            x = self.norm2(x)
            x = self.norm1(router(x, (H, W)) + x)
            x = layer(x, (H, W))

        out = self.norm(x)
        # Permute (B, H*W, D) -> (B, D, H, W)
        x = out.permute(0, 2, 1).view(B, D, H, W)
        return x

if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = GwinIR(
        upscale=2,
        img_size=(height, width),
        window_size=window_size,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffledirect')
    print(model)
    print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)
