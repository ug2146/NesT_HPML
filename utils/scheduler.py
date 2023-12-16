from torch.optim import lr_scheduler

def get_scheduler(key):
    if "linear" in key.lower(): return lr_scheduler.LinearLR
    if "cosine" in key.lower(): return lr_scheduler.CosineAnnealingLR