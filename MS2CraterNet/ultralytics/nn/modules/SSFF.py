import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f

class SSFF(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

    def forward(self, x):
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = F.interpolate(s, m.shape[2:], mode='nearest')
        lms = torch.cat([l, m, s], dim=1)
        return lms


if __name__ == '__main__':

    x = [torch.randn(4, 16, 256, 256),torch.randn(4, 16, 256, 256),torch.randn(4, 16, 256, 256)]
    SF = SSFF(160)
    output = SF(x)
    print(output.shape)