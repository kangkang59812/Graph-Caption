import os
import os.path as osp
import numpy as np
from time import strftime, localtime
from yacs.config import CfgNode as CN


_C = CN()


""""======================================="""
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1024
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1024
# Values to be used for image normalization
# [123.675,116.280,103.530]/255  ,RGB
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = False
_C.INPUT.BRIGHTNESS = 0.0          # Image ColorJitter
_C.INPUT.CONTRAST = 0.0            # Image ColorJitter
_C.INPUT.SATURATION = 0.0          # Image ColorJitter
_C.INPUT.HUE = 0.0                 # Image ColorJitter
_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0
""""======================================="""


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # 克隆一份配置节点_C的信息返回，_C的信息不会改变
    # This is for the "local variable" use pattern
    return _C.clone()
