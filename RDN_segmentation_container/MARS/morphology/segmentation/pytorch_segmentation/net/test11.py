from utils.dataset import HDF52D
import time
import utils.dataprocess as dp
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from unet import UNet
from utils.dice_loss import dice_coeff
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
batch_size = 1
c, h, w = 3, 10, 10
nb_classes = 5

x = torch.randn(batch_size, c, h, w)
target = torch.empty(batch_size, h, w, dtype=torch.long).random_(nb_classes)

model = nn.Sequential(
    nn.Conv2d(c, 6, 3, 1, 1),
    nn.ReLU(),
    nn.Conv2d(6, nb_classes, 3, 1, 1)
)

criterion = nn.CrossEntropyLoss()

output = model(x)
loss = criterion(output, target)
loss.backward()