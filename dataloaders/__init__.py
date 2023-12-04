from .cifar10 import Cifar10

def get_dataclass(config):
    if "cifar" in config['dataset'].lower() and "10" in config['dataset'].lower(): return Cifar10(config)