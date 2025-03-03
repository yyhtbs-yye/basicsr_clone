import torch
import torch.nn as nn
from basicsr.archs.arch_util import flow_warp

class BiRecurrentLayer2ndOrder(nn.Module):

    def __init__(self, in_chans, is_reversed, fextor):

        super().__init__()

        self.in_chans = in_chans
        self.is_reversed = is_reversed
        self.ch_aggr = nn.Conv2d(3*in_chans, in_chans, kernel_size=1)
        self.fextor = fextor # It should be an empty module
        self.warper = flow_warp

    def forward(self, curr_feats, flows):

        n, t, c, h, w = curr_feats.size()

        feat_indices = list(range(-1, -t - 1, -1)) \
                                if self.is_reversed \
                                    else list(range(t))

        history_feats = [curr_feats[:, feat_indices[0], ...], curr_feats[:, feat_indices[0], ...]]
        history_flows = [flows.new_zeros(n, 2, h, w), flows.new_zeros(n, 2, h, w)]

        out_feats = []

        for i in range(0, t):

            x = curr_feats[:, feat_indices[i], ...]
            y2, y1 = history_feats
            f2, f1 = history_flows
            a1 = self.warper(y1, f1.permute(0, 2, 3, 1))
            f2 = f1 + self.warper(f2, f1.permute(0, 2, 3, 1))
            a2 = self.warper(y2, f2.permute(0, 2, 3, 1))

            cond = torch.cat([a1, x, a2], axis=1) # concatenate at axis 1 (ch dim)

            cond = self.ch_aggr(cond)

            # x: current input; a: aligned prev, y: raw prev, dense
            o = self.fextor(cond) + cond

            out_feats.append(o)

            if i == t - 1: # for the last iter, need to to update history
                break

            # update history feats and flows
            history_feats = [history_feats[1], o]
            history_flows = [history_flows[1], flows[:, feat_indices[i], ...]]

        if self.is_reversed:
            out_feats = out_feats[::-1]

        if not self.training:  # model is in eval mode
            out_feats_cpu = []

            for feat in out_feats:
                out_feats_cpu.append(feat.cpu())  # Move to CPU
                del feat # Free GPU memory

            del history_feats, history_flows, y1, y2, a1, a2, o, cond
            torch.cuda.empty_cache()  # Optional: helps reclaim freed memory

            # Stack the CPU tensors
            stacked = torch.stack(out_feats_cpu, dim=1)

            # Move back to the original device (usually GPU)
            device = next(self.parameters()).device
            stacked = stacked.to(device)

            return stacked
        else:
            return torch.stack(out_feats, dim=1)