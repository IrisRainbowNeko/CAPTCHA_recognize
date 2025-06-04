from functools import partial

import torch
import torchvision.transforms as T
from rainbowneko.ckpt_manager import ckpt_saver
from rainbowneko.data import BaseDataset
from rainbowneko.data import SizeBucket
from rainbowneko.data.handler import HandlerChain, ImageHandler, LoadImageHandler
from rainbowneko.data.source import ImageLabelSource
from rainbowneko.evaluate import MetricGroup, MetricContainer, Evaluator
from rainbowneko.models.wrapper import SingleWrapper
from rainbowneko.parser import CfgWDModelParser, neko_cfg
from rainbowneko.train.loss import LossContainer
from rainbowneko.utils import CosineLR

from cfgs.py.train import train_base, tuning_base
from loss import ACE
from metrics import Precision, Recall, TextPreview
from models import ResnetEncoderDecoder

num_classes = 26+10
char_dict = '_0123456789abcdefghijklmnopqrstuvwxyz'

@neko_cfg
def make_cfg():
    return dict(
        _base_=[train_base, tuning_base],

        model_part=CfgWDModelParser([
            dict(
                lr=1e-4,
                layers=[''],  # train all layers
            )
        ]),

        # func(_partial_=True, ...) same as partial(func, ...)
        ckpt_saver=dict(
            model=ckpt_saver(target_module='model'),
            # optimizer=NekoOptimizerSaver(),
        ),

        train=dict(
            train_epochs=50,
            workers=4,
            max_grad_norm=10.,
            save_step=2000,

            loss=LossContainer(loss=ACE()),

            optimizer=partial(torch.optim.AdamW, weight_decay=1e-2),

            scale_lr=False,
            lr_scheduler=CosineLR(
                _partial_=True,
                warmup_steps=1000,
            ),
            metrics=MetricGroup(
                prcision=MetricContainer(Precision()),
                recall=MetricContainer(Recall())
            ),
        ),

        model=dict(
            name='ace-resnet18',
            wrapper=partial(SingleWrapper, model=ResnetEncoderDecoder(char_dict, num_classes))
        ),

        data_train=cfg_data(), # config can be split into another function with @neko_cfg

        evaluator=cfg_evaluator(),
    )

@neko_cfg
def cfg_data():
    return dict(
        dataset1=partial(BaseDataset, batch_size=256, loss_weight=1.0,
            source=dict(
                data_source1=ImageLabelSource(
                    img_root='',
                    label_file='',
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                bucket=SizeBucket.handler, # bucket 会自带一些处理模块
                image=ImageHandler(transform=T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]),
                )
            ),
            bucket=SizeBucket(num_bucket=1),
        )
    )

@neko_cfg
def cfg_evaluator():
    return partial(Evaluator,
        interval=1000,
        metrics=MetricGroup(
            prcision=MetricContainer(Precision()),
            recall=MetricContainer(Recall()),
            preview=TextPreview(char_dict)
        ),
        dataset=partial(BaseDataset, batch_size=128, loss_weight=1.0,
            source=dict(
                data_source1=ImageLabelSource(
                    img_root='',
                    label_file='',
                ),
            ),
            handler=HandlerChain(
                load=LoadImageHandler(),
                bucket=SizeBucket.handler,
                image=ImageHandler(transform=T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]),
                )
            ),
            bucket=SizeBucket(num_bucket=1),
        )
    )