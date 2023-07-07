""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
from .unet_parts import *
from .domian_enrich_block import DomainEnrich_Block

class MLP(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MLP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.inc = DoubleConv(n_channels, 64)
        self.out = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.inc(x)
        logits = self.out(x)


        return logits
