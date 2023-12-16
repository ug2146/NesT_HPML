import torch.nn as nn

def get_loss(key):
    if "ce" in key.lower(): return nn.CrossEntropyLoss()