#!/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.optim import SGD, Adam
import torch


def VIPMT_optimizer(model):
    """
    We recommend not adding memory (or use smaller learning rate) in the first 50 epochs for stability.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad],
            "lr": 5e-4,
            "betas": (0.9, 0.999),
            "weight_decay": 5e-4,
        }
    ]

    opt = Adam(param_dicts, lr=5e-4, betas=(0.9, 0.999), weight_decay=5e-4)
    return opt
