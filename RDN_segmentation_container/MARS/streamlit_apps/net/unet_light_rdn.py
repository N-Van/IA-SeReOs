""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
from .domian_enrich_block import DomainEnrich_Block

class UNet_Light_RDN(nn.Module):
    def __init__(self, n_channels, n_classes, dropout_rate, bilinear=True):
        super(UNet_Light_RDN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.rdn1 = DomainEnrich_Block(n_channels, 8)
        self.rdn2 = DomainEnrich_Block(n_channels, 8)

        # Change this line to use n_channels 
        self.inc = DoubleConv(n_channels, 32, dropout_rate)  # Adjust to accept 3-channel input
        
        self.down1 = Down(32, 64, dropout_rate)
        self.down2 = Down(64, 128, dropout_rate)
        self.down3 = Down(128, 256, dropout_rate)
        self.down4 = Down(256, 256, dropout_rate)
        self.up1 = Up(512, 128, dropout_rate, bilinear)
        self.up2 = Up(256, 64, dropout_rate, bilinear)
        self.up3 = Up(128, 32, dropout_rate, bilinear)
        self.up4 = Up(64, 32, dropout_rate, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)  # Now this will handle n-channel input
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

