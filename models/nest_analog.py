from .nest import nest_tiny, nest_small, nest_base

from aihwkit.nn import AnalogLinear
from aihwkit.nn import AnalogConv2d

from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.presets import ReRamESPreset

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import TikiTakaReRamSBPreset
from aihwkit.simulator.configs import MappingParameter

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

def get_rpuconfig():
    # mapping = MappingParameter(
    #     max_input_size=512,  # analog tile size
    #     max_output_size=512,
    #     digital_bias=True,
    #     weight_scaling_omega=0.6,
    #     )  # whether to use analog or digital bias
    
    # # Choose any preset or RPU configuration here
    # rpu_config = TikiTakaReRamSBPreset(mapping=mapping)
    # return rpu_config
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

def replace_linear_with_analog_linear(linear_layer, rpu_config):
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    bias = True if linear_layer.bias is not None else False
    analog_linear_layer = AnalogLinear(in_features, out_features, bias=bias, rpu_config=rpu_config)
    return analog_linear_layer

def convert_linear_to_analog_linear(model, rpu_config):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            # Replace Linear layer with AnalogLinear layer
            analog_linear_layer = replace_linear_with_analog_linear(layer, rpu_config)
            setattr(model, name, analog_linear_layer)
        elif list(layer.children()):
            # If the layer has children (e.g., nn.Sequential), recursively convert Linear layers
            convert_linear_to_analog_linear(layer, rpu_config)

def replace_conv2d_with_analog_conv2d(conv2d_layer, rpu_config):
    in_channels = conv2d_layer.in_channels
    out_channels = conv2d_layer.out_channels
    kernel_size = conv2d_layer.kernel_size
    stride = conv2d_layer.stride
    padding = conv2d_layer.padding
    
    analog_conv2d_layer = AnalogConv2d(in_channels, out_channels, kernel_size, stride, padding, rpu_config=rpu_config)
    return analog_conv2d_layer

def convert_conv2d_to_analog_conv2d(model, rpu_config):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            # Replace Conv2d layer with AnalogConv2d layer
            analog_conv2d_layer = replace_conv2d_with_analog_conv2d(layer, rpu_config)
            setattr(model, name, analog_conv2d_layer)
        elif list(layer.children()):
            # If the layer has children (e.g., nn.Sequential), recursively convert Conv2d layers
            convert_conv2d_to_analog_conv2d(layer, rpu_config)

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

def nest_tiny_analog(pretrained=False):
    """ Nest-T @ 224x224
    """
    model = nest_tiny(pretrained)
    print(model)
    rpu_config = get_rpuconfig()
    # convert_linear_to_analog_linear(model, rpu_config)
    # convert_conv2d_to_analog_conv2d(model, rpu_config)
    model = convert_to_analog(model, rpu_config)
    print(model)
    # exit()
    return model