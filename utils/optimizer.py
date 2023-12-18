from aihwkit.optim import AnalogAdam, AnalogSGD
from torch import optim


def get_optimizer(key):
    if "analog" in key.lower() and "adam" in key.lower(): return AnalogAdam
    if "analog" in key.lower() and "sgd" in key.lower(): return AnalogSGD
    if "adamw" in key.lower(): return optim.AdamW
    if "adam" in key.lower(): return optim.Adam
    if "sgd" in key.lower(): return optim.sgd

    
