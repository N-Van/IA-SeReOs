""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
from .unet_parts import *
from .domian_enrich_block import DomainEnrich_Block

class UNet_RDN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_RDN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.rdn1 = DomainEnrich_Block(n_channels, 32)
        self.rdn2 = DomainEnrich_Block(n_channels, 32)
        self.inc = DoubleConv(64, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.x_rdn1 = None
        self.x_rdn2 = None

    def forward(self, x):
        self.x_rdn1 = self.rdn1(x)
        self.x_rdn2 = self.rdn2(x)

        # self.x_rdn2 = self.rdn2(self.x_rdn1)
        x1 = self.inc(torch.cat((self.x_rdn2 ,self.x_rdn1), 1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
