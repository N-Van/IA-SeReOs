""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import numpy as np
from .unet_parts import *
from pytorch_wavelets import DWTForward
import torch

class UNet_Wavelet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, wave_conv_channels = 64):
        super(UNet_Wavelet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.avgpool = nn.AvgPool2d((2,2),stride = (2,2))
        self.down1 = Down(32, 64)
        self.down2 = Down(64 + wave_conv_channels, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128+ wave_conv_channels, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        # new structure for wavelet
        self.wave_transforms = DWTForward(J=4,wave='db1',mode='zero')
        self.delay = 1
        self.wave_conv1 = DoubleConv(4,wave_conv_channels)
        self.wave_conv2 = DoubleConv(4,wave_conv_channels)
        self.wave_conv3 = DoubleConv(4,wave_conv_channels)
        self.wave_conv4 = DoubleConv(4,wave_conv_channels)

    def forward(self, x):

        x_input1 = self.avgpool(x)
        x_input2 = self.avgpool(x_input1)
        x_input3 = self.avgpool(x_input2)
        x_input4 = self.avgpool(x_input3)

        input_size = np.asarray(list(x.shape[2:len(x.shape)]),dtype=int)
        input_size_wave1 = input_size / 2
        input_size_wave2 = input_size / 4
        input_size_wave3 = input_size / 8
        input_size_wave4 = input_size / 16
        x_wave = self.wave_transforms(x)
        x_wave1 = torch.squeeze(x_wave[1][0],1)
        x_wave1_shape = np.asarray(list(x_wave1.shape[2:len(x_wave1.shape)]),dtype=int)
        x_wave1 = x_wave1[:,:,int(x_wave1_shape[0] - input_size_wave1[0]):int(x_wave1_shape[0]),
                                  int(x_wave1_shape[1] - input_size_wave1[1]):int(x_wave1_shape[1])]
        x_wave2 = torch.squeeze(x_wave[1][1], 1)
        x_wave2_shape = np.asarray(list(x_wave2.shape[2:len(x_wave2.shape)]),dtype=int)
        x_wave2 = x_wave2[:,:,int(x_wave2_shape[0] - input_size_wave2[0]):int(x_wave2_shape[0]),
                                  int(x_wave2_shape[1] - input_size_wave2[1]):int(x_wave2_shape[1])]
        x_wave3 = torch.squeeze(x_wave[1][2], 1)
        x_wave3_shape = np.asarray(list(x_wave3.shape[2:len(x_wave3.shape)]),dtype=int)
        x_wave3 = x_wave3[:,:,int(x_wave3_shape[0] - input_size_wave3[0]):int(x_wave3_shape[0]),
                                  int(x_wave3_shape[1] - input_size_wave3[1]):int(x_wave3_shape[1])]
        x_wave4 = torch.squeeze(x_wave[1][3], 1)
        x_wave4_shape = np.asarray(list(x_wave4.shape[2:len(x_wave4.shape)]),dtype=int)
        x_wave4 = x_wave4[:,:,int(x_wave4_shape[0] - input_size_wave4[0]):int(x_wave4_shape[0]),
                                  int(x_wave4_shape[1] - input_size_wave4[1]):int(x_wave4_shape[1])]

        x_wave1 = self.wave_conv1(torch.cat((x_wave1,x_input1),1))
        x_wave2 = self.wave_conv1(torch.cat((x_wave2,x_input2),1))
        x_wave3 = self.wave_conv1(torch.cat((x_wave3,x_input3),1))
        x_wave4 = self.wave_conv1(torch.cat((x_wave4,x_input4),1))

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(torch.cat((x2,x_wave1),1))
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, torch.cat((x2,x_wave1),1))
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
