from nest import nest_tiny, nest_small, nest_base


def nest_base(pretrained=False, **kwargs) -> Nest:
    """ Nest-B @ 224x224
    """
    model = nest_tiny(pretrained)
    
    return model


def nest_small(pretrained=False, **kwargs) -> Nest:
    """ Nest-S @ 224x224
    """
    model_kwargs = dict(img_size=32, embed_dims=(96, 192, 384), num_heads=(3, 6, 12), depths=(2, 2, 20), **kwargs)
    model = _create_nest('nest_small', pretrained=pretrained, **model_kwargs)
    return model

def nest_tiny(pretrained=False, **kwargs) -> Nest:
    """ Nest-T @ 224x224
    """
    model_kwargs = dict(img_size=32, embed_dims=(96, 192, 384), num_heads=(3, 6, 12), depths=(2, 2, 8), **kwargs)
    model = _create_nest('nest_tiny', pretrained=pretrained, **model_kwargs)
    return model