import torch.nn as nn
import torch

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DomainEnrich_Block(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.basic_block1 = BasicBlock(n_channels, n_classes)
        self.basic_block2 = BasicBlock(n_classes, n_classes)

    def forward(self, x):
        
        x = self.basic_block1(x)
        x = self.basic_block2(x)
        
        return x

class RDN_Block(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.rdn1 = DomainEnrich_Block(n_channels, n_classes)
        self.rdn2 = DomainEnrich_Block(n_channels, n_classes)

    def forward(self, x):
        self.x_rdn1 = self.rdn1(x)
        self.x_rdn2 = self.rdn2(x)
        x = torch.cat((self.x_rdn2, self.x_rdn1), 1)
        # self.x_rdn2 = self.rdn2(self.x_rdn1)
        return x