from .nest import nest_tiny, nest_small, nest_base

def get_model(key, pretrained = False):
    if "nest_t" or "nest-t" in key.lower(): return nest_tiny(pretrained)
    if "nest_s" or "nest-s" in key.lower(): return nest_small(pretrained)
    if "nest_b" or "nest-b" in key.lower(): return nest_base(pretrained)