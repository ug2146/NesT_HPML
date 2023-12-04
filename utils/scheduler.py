from pl_bolts.optimizer import lr_scheduler

def get_scheduler(key):
    if "linearwarmupcosineannealing" in key.lower(): return lr_scheduler.LinearWarmupCosineAnnealingLR