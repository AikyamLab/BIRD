import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import json
from flopth import flopth


def all_ones_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.eye_(m.weight)
        nn.init.zeros_(m.bias)


class ChannelPruningv2(nn.Module):
    def __init__(self, config, in_channels):
        super(ChannelPruningv2, self).__init__()
        self.param = nn.Parameter(torch.ones(1, in_channels))
        self.config = config

    def forward(self, z):
        if self.config.sigmoid:
            b = torch.sigmoid(self.param)
        else:
            b = self.param
                        
        return z * b
        

if __name__ == "__main__":
    x = torch.randn((1, 512), requires_grad=True).cuda()
    net = ChannelPruningv2(512).cuda()

    print(flopth(net, x.shape))

    out = net(x)
    out.sum().backward()
    print(out.mean())
    print(x.grad.mean())
