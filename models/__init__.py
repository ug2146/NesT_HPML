from .nest import nest_tiny, nest_small, nest_base
from .nest_analog import nest_tiny_analog, nest_small_analog, nest_base_analog
from .nest_msa import nest_msa_tiny, nest_msa_small, nest_msa_base
from .resnet_analog import resnet34

def get_model(key, pretrained = False, preset=None):

    if "analog" in key.lower() and "resnet" in key.lower(): return resnet34(preset)

    if "nest_t" in key.lower() or "nest-t" in key.lower() and "analog" in key.lower(): return nest_tiny_analog(pretrained, preset)
    if "nest_s" in key.lower() or "nest-s" in key.lower() and "analog" in key.lower(): return nest_small_analog(pretrained)
    if "nest_b" in key.lower() or "nest-b" in key.lower() and "analog" in key.lower(): return nest_base_analog(pretrained)
    
    if "nest_t" in key.lower() or "nest-t" in key.lower(): return nest_tiny(pretrained)
    if "nest_s" in key.lower() or "nest-s" in key.lower(): return nest_small(pretrained)
    if "nest_b" in key.lower() or "nest-b" in key.lower(): return nest_base(pretrained)
    # nest model with MSA: 
    if "nest_msa_t" in key.lower() or "nest_msa-t" in key.lower(): return nest_msa_tiny(pretrained)
    if "nest_msa_s" in key.lower() or "nest_msa-s" in key.lower(): return nest_msa_small(pretrained)
    if "nest_msa_b" in key.lower() or "nest_msa-b" in key.lower(): return nest_msa_base(pretrained)
    

    