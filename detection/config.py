# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import os.path as osp
import numpy as np
from time import strftime, localtime
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# 目标检测
__C.DATA_DIR = '/home/lkk/code/my_faster/datasets/vg/VG'
