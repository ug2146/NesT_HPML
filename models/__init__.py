from .nest import nest_tiny, nest_small, nest_base
from .NesT_MSA import nest_MSA_tiny, nest_MSA_small, nest_MSA_base

def get_model(key, pretrained = False):
    if "nest_t" in key.lower() or "nest-t" in key.lower(): return nest_tiny(pretrained)
    if "nest_s" in key.lower() or "nest-s" in key.lower(): return nest_small(pretrained)
    if "nest_b" in key.lower() or "nest-b" in key.lower(): return nest_base(pretrained)
    # nest model with MSA: 
    if "nest_msa_t" in key.lower() or "nest_msa-t" in key.lower(): return nest_MSA_tiny(pretrained)
    if "nest_msa_s" in key.lower() or "nest_msa-s" in key.lower(): return nest_MSA_small(pretrained)
    if "nest_msa_b" in key.lower() or "nest_msa-b" in key.lower(): return nest_MSA_base(pretrained)
    