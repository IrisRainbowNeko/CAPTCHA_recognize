import torch
from torch import nn


class ACE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, label):
        B, H, W, C = input.size()
        T_ = H * W

        input = input.view(B, T_, -1)
        input = input + 1e-10

        label[:, 0] = T_ - label[:, 0]

        # ACE Implementation (four fundamental formulas)
        input = torch.sum(input, 1)
        input = input / T_
        label = label / T_

        loss = (-torch.sum(torch.log(input) * label)) / B

        return loss
