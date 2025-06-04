# -*- coding: utf-8 -*-

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetEncoderDecoder(nn.Module):
    def __init__(self, char_dict, class_num=26+10):
        super(ResnetEncoderDecoder, self).__init__()
        self.bn = nn.BatchNorm2d(64)
        resnet = timm.create_model('resnet18', pretrained=True, drop_rate=0.2, drop_path_rate=0.3)
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.cnn = nn.Sequential(*list(resnet.children())[4:-2])
        self.out = nn.Linear(512, class_num + 1)

        self.char_dict = char_dict

    def forward(self, input, labels):
        input = F.silu(self.bn(self.conv(input)), True)
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
        input = self.cnn(input)

        input = input.permute(0, 2, 3, 1)
        input = F.softmax(self.out(input), dim=-1)

        if labels is not None:
            return input
        else:
            self.bs, self.h, self.w, _ = input.size()
            T_ = self.h * self.w
            input = input.view(self.bs, T_, -1)
            input = input + 1e-10

            pred = torch.max(input, 2)[1].data.cpu().numpy()
            pred = pred[0]  # sample #0

            pred_string = ''.join(['%2s' % self.char_dict[pn] for pn in pred])

            pred_string_set = [pred_string[i:i + self.w * 2] for i in range(0, len(pred_string), self.w * 2)]
            print('Prediction: ')
            for pre_str in pred_string_set:
                print(pre_str)

            print(self.w, self.h)
            print(pred)
            return pred.reshape((self.h, self.w)).T.reshape((self.h * self.w,))
