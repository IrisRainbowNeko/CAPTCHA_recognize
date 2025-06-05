from torch import nn
import torch
from torch.nn import functional as F
from rainbowneko.evaluate import BaseMetric
from rainbowneko.utils import KeyMapper
from utils import remove_rptch
from einops import rearrange


class Precision(nn.Module):
    def forward(self, pred, label):
        '''

        :param pred: [B,H*W,C]
        :param label: [B,C]
        :return:
        '''
        pred = rearrange(pred, 'b h w c -> b (h w) c')

        total_char_count = label[:, 1:].sum()
        pred_chars = pred.argmax(dim=2)
        result = F.one_hot(pred_chars, pred.size(2)).float().sum(dim=1)
        FN_count = torch.relu(label[:, 1:] - result[:, 1:]).sum()
        total_pred_count = pred_chars.count_nonzero()
        correct_count = total_char_count - FN_count

        return correct_count / (total_pred_count + 0.000001)


class Recall(nn.Module):
    def forward(self, pred, label):
        '''

        :param pred: [B,H*w,C]
        :param label: [B,C]
        :return:
        '''
        pred = rearrange(pred, 'b h w c -> b (h w) c')

        total_char_count = label[:, 1:].sum()
        pred_chars = pred.argmax(dim=2)
        result = F.one_hot(pred_chars, pred.size(2)).float().sum(dim=1)
        FN_count = torch.relu(label[:, 1:] - result[:, 1:]).sum()
        correct_count = total_char_count - FN_count

        return float(correct_count) / total_char_count

class TextPreview(BaseMetric):
    def __init__(self, char_dict, device='cpu', key_map=None):
        super().__init__()
        self.key_mapper = KeyMapper(key_map=key_map or KeyMapper.cls_map)
        self.device = device
        self.char_dict = char_dict

    def reset(self):
        self.metric = None

    def make_preview(self, pred, label):
        B, H, W, C = pred.size()
        T_ = H * W
        pred = pred.view(B, T_, -1)
        pred = pred + 1e-10

        pred = torch.max(pred, 2)[1].data.cpu().numpy()
        pred = pred[0]
        label_i = label[0].cpu()
        label_string = ' '.join([f'{self.char_dict[i]}:{li}' for i, li in enumerate(label_i) if li > 0])

        pred_string = ''.join(['%2s' % self.char_dict[pn] for pn in pred])

        pred_string_set = [pred_string[i:i + W * 2] for i in range(0, len(pred_string), W * 2)]
        print('Prediction: ')
        for pre_str in pred_string_set:
            print(pre_str)

        print(W, H)
        print(pred)
        print('Label:', label_string)
        pred = pred.reshape((H, W)).T.reshape((H * W,))
        final_str = remove_rptch(''.join(self.char_dict[x] for x in pred if x))
        print(final_str)
        return final_str

    def update(self, pred, inputs):
        if self.metric is None:
            args, kwargs = self.key_mapper(pred=pred, inputs=inputs)
            v_metric = self.make_preview(*args, **kwargs)
            self.metric = v_metric

    def finish(self, gather, is_local_main_process):
        return self.metric

    def to(self, device):
        self.device = device