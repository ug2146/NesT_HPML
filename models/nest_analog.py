from .nest import nest_tiny, nest_small, nest_base

from utils.rpuconfig import get_rpuconfig

from aihwkit.nn import AnalogLinear
from aihwkit.nn import AnalogConv2d
from aihwkit.nn.conversion import convert_to_analog

from aihwkit.simulator.configs import (
    RPUDataType,
    InferenceRPUConfig,
    WeightRemapType,
    WeightModifierType,
    WeightClipType,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.inference import PCMLikeNoiseModel

import torch.nn as nn


def get_default_rpuconfig():
    my_rpu_config = InferenceRPUConfig()
    my_rpu_config.mapping.digital_bias = True
    my_rpu_config.mapping.out_scaling_columnwise = True
    my_rpu_config.mapping.learn_out_scaling = True
    my_rpu_config.mapping.weight_scaling_omega = 1.0
    my_rpu_config.mapping.weight_scaling_columnwise = False
    my_rpu_config.mapping.max_input_size = 512
    my_rpu_config.mapping.max_output_size = 512

    my_rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    my_rpu_config.remap.type = WeightRemapType.CHANNELWISE_SYMMETRIC
    my_rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    my_rpu_config.clip.sigma = 2.5

    # train input clipping
    my_rpu_config.forward.noise_management = NoiseManagementType.NONE
    my_rpu_config.forward.bound_management = BoundManagementType.NONE
    my_rpu_config.forward.out_bound = 10.0  # quite restrictive
    my_rpu_config.pre_post.input_range.enable = True
    my_rpu_config.pre_post.input_range.manage_output_clipping = True
    my_rpu_config.pre_post.input_range.decay = 0.001
    my_rpu_config.pre_post.input_range.input_min_percentage = 0.95
    my_rpu_config.pre_post.input_range.output_min_percentage = 0.95

    my_rpu_config.modifier.type = WeightModifierType.ADD_NORMAL
    my_rpu_config.modifier.std_dev = 0.1

    return my_rpu_config

def nest_base_analog(pretrained=False):
    """ Nest-B @ 224x224
    """
    model = nest_base(pretrained)
    return model

def nest_small_analog(pretrained=False):
    """ Nest-S @ 224x224
    """
    model = nest_small(pretrained)
    return model

def nest_tiny_analog(pretrained=False, key=None):
    """ Nest-T @ 224x224
    """
    model = nest_tiny(pretrained)
    # print(model)
    if key != None:
        rpu_config = get_rpuconfig(key)
        rpu_config = rpu_config()
    else:
        rpu_config = get_default_rpuconfig()
    
    model = convert_to_analog(model, rpu_config)
    print(model)
    # exit()
    return model