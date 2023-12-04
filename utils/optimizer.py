from torch import optim 

def get_optimizer(key):
    if "adamw" in key.lower(): return optim.AdamW
    if "adam" in key.lower(): return optim.Adam
    if "sgd" in key.lower(): return optim.sgd
    
