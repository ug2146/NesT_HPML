from .nest import nest_tiny, nest_small, nest_base
from .nest_analog import nest_tiny_analog, nest_small_analog, nest_base_analog
from .NesT_MSA import nest_MSA_tiny, nest_MSA_small, nest_MSA_base

def get_model(key, pretrained = False, preset=None):

    if "nest_t" in key.lower() or "nest-t" in key.lower() and "analog" in key.lower(): return nest_tiny_analog(pretrained, preset)
    if "nest_s" in key.lower() or "nest-s" in key.lower() and "analog" in key.lower(): return nest_small_analog(pretrained)
    if "nest_b" in key.lower() or "nest-b" in key.lower() and "analog" in key.lower(): return nest_base_analog(pretrained)
    
    if "nest_t" in key.lower() or "nest-t" in key.lower(): return nest_tiny(pretrained)
    if "nest_s" in key.lower() or "nest-s" in key.lower(): return nest_small(pretrained)
    if "nest_b" in key.lower() or "nest-b" in key.lower(): return nest_base(pretrained)
    # nest model with MSA: 
    if "nest_msa_t" in key.lower() or "nest_msa-t" in key.lower(): return nest_MSA_tiny(pretrained)
    if "nest_msa_s" in key.lower() or "nest_msa-s" in key.lower(): return nest_MSA_small(pretrained)
    if "nest_msa_b" in key.lower() or "nest_msa-b" in key.lower(): return nest_MSA_base(pretrained)
    

    