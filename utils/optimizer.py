from aihwkit.optim import AnalogAdam
from torch import optim


def get_optimizer(key):
    if "analog" in key.lower(): return AnalogAdam
    if "adamw" in key.lower(): return optim.AdamW
    if "adam" in key.lower(): return optim.Adam
    if "sgd" in key.lower(): return optim.sgd

    
