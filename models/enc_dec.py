# -*- coding: utf-8 -*-

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetEncoderDecoder(nn.Module):
    def __init__(self, char_dict):
        super(ResnetEncoderDecoder, self).__init__()
        self.bn = nn.BatchNorm2d(64)
        resnet = timm.create_model('resnet18', pretrained=True, drop_rate=0.2, drop_path_rate=0.3)
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.cnn = nn.Sequential(*list(resnet.children())[4:-2])
        self.out = nn.Linear(512, len(char_dict))

        self.char_dict = char_dict

    def forward(self, input):
        input = F.silu(self.bn(self.conv(input)), True)
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
        input = self.cnn(input)

        input = input.permute(0, 2, 3, 1)
        input = F.softmax(self.out(input), dim=-1)

        return input
