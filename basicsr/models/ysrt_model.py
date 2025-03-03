import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.video_recurrent_model import VideoRecurrentModel

@MODEL_REGISTRY.register()
class YsrtModel(VideoRecurrentModel):

    def test(self):
        n = self.lq.size(1)
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        img = self.lq

        if 'window_size' in self.opt['plugin']['net_para']:
            window_size = self.opt['plugin']['net_para']['window_size']
            scale = self.opt.get('scale', 1)
            mod_pad_h, mod_pad_w = 0, 0
            _, _, _, h, w = self.lq.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            lq_padded = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h, 0, 0), mode='constant', value=0)
            # Manually reflect the values instead of using 'reflect' mode
            if mod_pad_h > 0:
                lq_padded[:, :, :, -mod_pad_h:, :] = lq_padded[:, :, :, -mod_pad_h-1:-1, :]
            if mod_pad_w > 0:
                lq_padded[:, :, :, :, -mod_pad_w:] = lq_padded[:, :, :, :, -mod_pad_w-1:-1]
            img = lq_padded

        with torch.no_grad():
            self.output = self.net_g(img)

        if 'window_size' in self.opt['plugin']['net_para']:
            _, _, _, h, w = self.output.size()
            self.output = self.output[:, :, :, 0 : h - mod_pad_h * scale,
                                               0 : w - mod_pad_w * scale]

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()
