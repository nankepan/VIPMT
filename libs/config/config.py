#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from easydict import EasyDict
OPTION = EasyDict()

OPTION.data_path = "data/"  # path to dataset
OPTION.initmodel_path = "pretrain_model/"  # path to ImageNet pretrained backbone
OPTION.input_size = (240, 424)
OPTION.test_size = (240, 424)
OPTION.SNAPSHOT_DIR = 'results/'
