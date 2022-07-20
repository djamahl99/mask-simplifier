from functools import partial
from re import M, X
from tkinter.tix import MAX
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from generate_dataset import MAX_SEQ_LEN
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=bias),
        nn.ReLU(inplace=True),
    )

class PolygonPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # filters = [256, 512, 1024, 512, 256]
        # filters = [64, 128, 256, 128, 64]
        filters = [32, 64, 128, 64, 32]

        bias = False

        self.conv1 = convrelu(1, filters[0], 3, 1, bias=bias)
        self.l1 = convrelu(filters[0], 1, 1, 0)
        self.batch_norm1 = nn.BatchNorm2d(filters[0])

        self.conv2 = convrelu(filters[0], filters[1], 3, 1, bias=bias)
        self.l2 = convrelu(filters[1], 1, 1, 0)
        self.batch_norm2 = nn.BatchNorm2d(filters[1])
        
        self.conv3 = convrelu(filters[1], filters[2], 3, 1, bias=bias)
        self.l3 = convrelu(filters[2], 1, 1, 0)
        self.batch_norm3 = nn.BatchNorm2d(filters[2])
        
        self.conv4 = convrelu(filters[2], filters[3], 3, 1, bias=bias)
        self.l4 = convrelu(filters[3], 1, 1, 0)
        self.batch_norm4 = nn.BatchNorm2d(filters[3])
        
        self.conv5 = convrelu(filters[3], filters[4], 3, 1, bias=bias)
        self.batch_norm5 = nn.BatchNorm2d(filters[4])

        self.out = nn.Sequential(
            nn.Conv2d(filters[4], 1, 1, 1),
            nn.Sigmoid()
            # nn.ReLU(True)
            # nn.Softmax2d()
        )

        self.angle = nn.Sequential(
            nn.Conv2d(filters[4], 1, 3, 1, 1),
            nn.ReLU(True)
        )

        self.len_pooling = nn.Sequential(
            nn.MaxPool2d(3, 2), # 112
            nn.AvgPool2d(3, 2), # 56
            nn.MaxPool2d(3, 2), # 28
            nn.AvgPool2d(3, 2), # 14
            # conv3x3(1, 16, 2),
            # conv3x3(16, 16, 2),
            # conv3x3(16, 16, 2),
            # conv3x3(16, 1, 2)
        )

        self.len_head = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, MAX_SEQ_LEN),
            nn.Sigmoid()
        )

        self.maxpool = nn.MaxPool2d(5, 2, 2, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(5, 2, 2)

    def forward(self, input):
        x1 = self.conv1(input)
        x1 = x1 + input
        x1 = self.batch_norm1(x1)

        x2 = self.conv2(x1)
        x2 = x2 + self.l1(x1)
        x2 = self.batch_norm2(x2)

        x3 = self.conv3(x2)
        x3 = x3 + self.l2(x2)
        x3 = self.batch_norm3(x3)

        x4 = self.conv4(x3)
        x4 = x4 + self.l3(x3)
        x4 = self.batch_norm4(x4)

        x5 = self.conv5(x4)
        x5 = x5 + self.l4(x4)
        x5 = self.batch_norm5(x5)

        out = self.out(x5)
        angle = self.angle(x5)

        out, indices = self.maxpool(out)
        out = self.maxunpool(out, indices, output_size=input.size())

        pool = self.len_pooling(out)
        pool = torch.flatten(pool, 1)
        length = self.len_head(pool)

        return length, out, angle