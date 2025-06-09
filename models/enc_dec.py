# -*- coding: utf-8 -*-

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from timm.models import resnet
from timm.layers import LayerNorm2d, LayerNorm

class LayerNorm2dNoBias(LayerNorm2d):
    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels, **kwargs)
        self.eps = kwargs.get('eps', 1e-6)
        self.bias = None

class MlpHead(nn.Module):
    """ MLP classification head
    """

    def __init__(
            self,
            dim,
            num_classes=1000,
            mlp_ratio=4,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm,
            drop_rate=0.,
            bias=True
    ):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_drop(x)
        x = self.fc2(x)
        return x

class ResnetEncoderDecoder(nn.Module):
    def __init__(self, char_dict, drop_rate=0.2, drop_path_rate=0.3):
        super(ResnetEncoderDecoder, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            LayerNorm2dNoBias(64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        # self.stem = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(64),
        #     nn.SiLU(),
        #     nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        # )

        resnet = timm.create_model('resnet18.fb_swsl_ig1b_ft_in1k', pretrained=True, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        self.cnn = nn.Sequential(*list(resnet.children())[4:-2])

        self.head = nn.Sequential(OrderedDict([
            ('norm', LayerNorm(512)),
            ('fc', MlpHead(512, len(char_dict))),
        ]))
        # self.head = nn.Sequential(
        #     nn.Linear(512, len(char_dict))
        # )

        self.char_dict = char_dict

    def forward(self, input):
        input = self.stem(input)
        input = self.cnn(input)

        input = input.permute(0, 2, 3, 1)
        input = F.softmax(self.head(input), dim=-1)

        return input

# class ResnetEncoderDecoder(nn.Module):
#     def __init__(self, char_dict, drop_rate=0.2, drop_path_rate=0.3):
#         super(ResnetEncoderDecoder, self).__init__()
#         self.bn = nn.BatchNorm2d(64)
#         resnet = timm.create_model('resnet18.tv_in1k', pretrained=True, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
#         self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
#         self.cnn = nn.Sequential(*list(resnet.children())[4:-2])
#         self.out = nn.Linear(512, len(char_dict))

#         self.char_dict = char_dict

#     def forward(self, input):
#         input = F.silu(self.bn(self.conv(input)), True)
#         input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
#         input = self.cnn(input)

#         input = input.permute(0, 2, 3, 1)
#         input = F.softmax(self.out(input), dim=-1)

#         return input

class CaformerEncoderDecoder(nn.Module):
    def __init__(self, char_dict, drop_rate=0.2, drop_path_rate=0.3):
        super().__init__()
        self.bn = nn.BatchNorm2d(64)
        backbone = timm.create_model('caformer_s18.sail_in22k_ft_in1k', pretrained=True, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        backbone.set_grad_checkpointing(True)
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.cnn = nn.Sequential(*list(backbone.children())[1:-1])
        self.out = nn.Linear(512, len(char_dict))

        self.char_dict = char_dict

    def forward(self, input):
        input = F.silu(self.bn(self.conv(input)), True)
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
        input = self.cnn(input)

        input = input.permute(0, 2, 3, 1)
        input = F.softmax(self.out(input), dim=-1)

        return input