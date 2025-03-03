import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.virtual_tbb import BiRecurrentLayer2ndOrder
from basicsr.archs.spynet_arch import SpyNet
from basicsr.archs.arch_util import Upsample

from basicsr.archs.basicvsr_arch import ConvResidualBlocks
@ARCH_REGISTRY.register()
class YsrtNet(nn.Module):

    def __init__(self, in_chans=3, embed_dim=180, mid_chans=64, out_chans=64, scale=2, spynet_pretrained=None):

        super().__init__()

        self.embed_dim = embed_dim
        self.scale = scale

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.conv_after_body = nn.Conv2d(mid_chans, out_chans, 3, 1, 1)
        self.conv_last = nn.Conv2d(out_chans, in_chans, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.pre_align = nn.Conv2d(embed_dim, mid_chans, 3, 1, 1)

        # the feature extractors that would be plugin from pretrained SISR
        self.pre_fextor = nn.Identity()
        self.main_fextor = nn.Sequential(
            nn.Conv2d(mid_chans, mid_chans, 1),
            ConvResidualBlocks(mid_chans, mid_chans, 6)
        )

        self.post_fextor = nn.Identity()

        # Recurrent propagators
        self.backward_propagator1 = BiRecurrentLayer2ndOrder(mid_chans, is_reversed=True, fextor=self.main_fextor)
        self.forward_propagator1  = BiRecurrentLayer2ndOrder(mid_chans, is_reversed=False, fextor=self.main_fextor)
        self.backward_propagator2 = BiRecurrentLayer2ndOrder(mid_chans, is_reversed=True, fextor=self.main_fextor)
        self.forward_propagator2  = BiRecurrentLayer2ndOrder(mid_chans, is_reversed=False, fextor=self.main_fextor)

        self.upsmaple = Upsample(scale, mid_chans)

        # optical flow network for feature alignment
        self.spynet = SpyNet(load_path=spynet_pretrained)

    def compute_flow(self, lrs):

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        backward_flows = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)
        forward_flows = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)

        return forward_flows, backward_flows

    def forward_feature(self, x, backward_flows, forward_flows):

        x = self.backward_propagator1(x, backward_flows)

        x = self.forward_propagator1(x, forward_flows)

        x = self.backward_propagator2(x, backward_flows)

        x = self.forward_propagator2(x, forward_flows)

        return x

    def forward(self, lrs):

        n, t, c, h, w = lrs.size()
        nt = n*t
        sh, sw = h*self.scale, w*self.scale

        # compute optical flow
        forward_flows, backward_flows = self.compute_flow(lrs)

        feats_ = self.conv_first(lrs.view(-1, c, h, w))

        if not self.training:  # model is in eval mode
            # Reshape feats_ from (n*t, embed_dim, h, w) to (n, t, embed_dim, h, w)
            feats_ = feats_.view(n, t, -1, h, w)
            feats__frames = []
            for i in range(t):
                # Process each frame separately; each frame has shape (n, embed_dim, h, w)
                frame_feats = feats_[:, i]
                processed_frame = self.pre_fextor(frame_feats)
                feats__frames.append(processed_frame)
            # Stack frames back along time dimension and flatten to (n*t, embed_dim, h, w)
            feats_ = torch.stack(feats__frames, dim=1).view(n * t, -1, h, w)

            del processed_frame, frame_feats, feats__frames

            torch.cuda.empty_cache()

        else:
            # In training mode, process all frames at once.
            feats_ = self.pre_fextor(feats_.view(n*t, -1, h, w)).view(n, t, -1, h, w)

        feats_ = self.pre_align(feats_.view(n*t, -1, h, w)).view(n, t, -1, h, w)

        feats_ = self.forward_feature(feats_.view(n, t, -1, h, w), backward_flows, forward_flows).view(nt, -1, h, w)

        featso = self.lrelu(self.conv_after_body(feats_) + feats_)

        featsu = self.post_fextor(featso)

        srs = self.conv_last(self.upsmaple(featsu)).view(n, t, -1, sh, sw)

        brs = F.interpolate(lrs.view(-1, c, h, w), scale_factor=self.scale, mode='bilinear',
                            align_corners=False).view(n, t, -1, sh, sw)

        return srs + brs

if __name__ == '__main__':
    tensor_filepath = "test_input_tensor2_7_3_64_64.pt"
    input_tensor = torch.load(tensor_filepath) / 100
    model = YsrtNet(in_chans=3, embed_dim=4,
                    spynet_pretrained='experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth')

    output1 = model(input_tensor)
