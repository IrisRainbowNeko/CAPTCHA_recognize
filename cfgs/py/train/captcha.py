from functools import partial

import torch
import torchvision.transforms as T
import numpy as np
from rainbowneko.ckpt_manager import ckpt_saver
from rainbowneko.data import BaseDataset
from rainbowneko.data import SizeBucket
from rainbowneko.loggers import CLILogger
from rainbowneko.data.handler import HandlerChain, ImageHandler, LoadImageHandler, DataHandler
from rainbowneko.data.source import ImageLabelSource
from rainbowneko.evaluate import MetricGroup, MetricContainer, Evaluator
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.parser import CfgWDModelParser, neko_cfg
from rainbowneko.train.loss import LossContainer
from rainbowneko.utils import CosineLR, ConstantLR

from cfgs.py.train import train_base, tuning_base
from loss import ACE
from metrics import Precision, Recall, TextPreview
from models import ResnetEncoderDecoder

char_dict = '_0123456789abcdefghijklmnopqrstuvwxyz'
data_root = '/data1/dzy/CAPTCHA_recognize/data_pp'

@neko_cfg
def make_cfg():
    return dict(
        _base_=[train_base, tuning_base],
        exp_dir='exps/captcha_pp/',

        model_part=CfgWDModelParser([
            dict(
                lr=2e-4,
                layers=[''],  # train all layers
            )
        ]),

        # func(_partial_=True, ...) same as partial(func, ...)
        ckpt_saver=dict(
            model=ckpt_saver(target_module='model'),
            # optimizer=NekoOptimizerSaver(),
        ),

        train=dict(
            train_steps=5000,
            # train_epochs=3,
            workers=4,
            max_grad_norm=10.,
            save_step=200,

            loss=LossContainer(loss=ACE()),

            optimizer=partial(torch.optim.AdamW, weight_decay=1e-2),

            scale_lr=False,
            lr_scheduler=ConstantLR(
                _partial_=True,
                warmup_steps=0,
            ),
            metrics=MetricGroup(
                prcision=MetricContainer(Precision()),
                recall=MetricContainer(Recall())
            ),
        ),

        model=dict(
            name='ace-resnet18',
            wrapper=partial(SingleWrapper, model=ResnetEncoderDecoder(char_dict))
        ),

        logger=[
            partial(CLILogger, out_path='train.log', log_step=20),
        ],

        data_train=cfg_data(), # config can be split into another function with @neko_cfg

        evaluator=cfg_evaluator(),
    )

class LabelHandler(DataHandler):
    def __init__(self, key_map_in=('label -> label',), key_map_out=('label -> label',)):
        super().__init__(key_map_in, key_map_out)
        self.cls_map = {c: i for i, c in enumerate(char_dict[1:])}
        self.class_num = len(char_dict)

    def handle(self, label):
        word = [self.cls_map[var] for var in label]
        label_t = np.zeros((self.class_num)).astype('float32')

        for ln in word:
            label_t[int(ln+1)] += 1 # label construction for ACE
        # print(f'word: {word}, label: {label}, {label_t}')

        label_t[0] = len(word)
        return {'label': label_t}

@neko_cfg
def cfg_data():
    return dict(
        dataset1=partial(BaseDataset, batch_size=256, loss_weight=1.0,
            source=dict(
                data_source1=ImageLabelSource(
                    img_root=data_root,
                    label_file=f'{data_root}/train.json',
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                bucket=SizeBucket.handler, # bucket 会自带一些处理模块
                image=ImageHandler(transform=T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]),
                ),
                label=LabelHandler(),
            ),
            bucket=SizeBucket.from_files(
                num_bucket=3*4,
                step_size=5,
                pre_build_bucket=f'{data_root}/train_bucket.pkl'
            ),
        )
    )

@neko_cfg
def cfg_evaluator():
    return partial(Evaluator,
        interval=100,
        metric=MetricGroup(
            prcision=MetricContainer(Precision()),
            recall=MetricContainer(Recall()),
            preview=TextPreview(char_dict)
        ),
        dataset=partial(BaseDataset, batch_size=128, loss_weight=1.0,
            source=dict(
                data_source1=ImageLabelSource(
                    img_root=data_root,
                    label_file=f'{data_root}/test.json',
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                bucket=SizeBucket.handler,
                image=ImageHandler(transform=T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]),
                ),
                label=LabelHandler(),
            ),
            bucket=SizeBucket.from_files(num_bucket=3*4, step_size=5),
        )
    )