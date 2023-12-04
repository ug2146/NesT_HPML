# Baseline Nested Hierarchical Transformer training

import argparse
import logging
import os
import yaml

import torch

from models import get_model
from dataloaders import get_dataclass


from pprint import pprint
from utils.logging import config_logging
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler


def argument_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/default.yaml', help='YAML configuration file')
    add_arg('--name', help='name of the experiment. Also used as output directory')
    add_arg('--model', help='model to be used')
    add_arg('--dataset', help='dataset to be used')
    add_arg('--train-batch-size', '-train-bs', help='training batch size')
    add_arg('--val-batch-size', '-val-bs', help='validation batch size')
    add_arg('--lr', help='learning rate for the training')
    add_arg('--optimizer', help='optimizer for the training')
    add_arg('--num-workers', help='number of workers')
    add_arg('--ngpus', help='number of gpus to be used')
    add_arg('--resume', action='store_true', help='Resume training from last checkpoint')
    add_arg('-v', '--verbose', action='store_true', help='Enable verbose logging')


    return parser.parse_args()

def load_config(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    for arg in vars(args):
        if arg != 'config' and getattr(args, arg) != None:
            config[arg] = getattr(args, arg)
    
    return config


def main(config):

    pprint(config)

    # Support for only single-gpu training
    assert(config['ngpus'] == 1, "Sorry! Only single-gpu training is supported!")
    
    # Create output/experiment folder and save config file
    output_dir = os.path.join("experiments", config['name'])
    os.makedirs(output_dir, exist_ok = True)
    with open(os.path.join(output_dir, config.yaml), 'w') as outfile:
        yaml.dump(config, outfile, sort_keys=False, default_flow_style=False)

    # Setup logging
    log_file = os.path.join(output_dir, 'info.log')
    config_logging(verbose=config['verbose'], log_file=log_file, append=config['resume'])
    logging.info(f"Initialized the logging for experiment: {config['name']}")

    # Initialize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataClass = get_dataclass(config)
    
    model = get_model(config['model'], pretrained=False)
    model.to(device)

    train_dl = dataClass.make_dataloader(train=True)
    test_dl = dataClass.make_dataloader(train=False)

    optimizer = get_optimizer(config['optimizer'])
    optimizer = optimizer(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    scheduler = get_scheduler(config['scheduler'])
    scheduler = scheduler(optimizer, warmup_epochs=config['warmup_epochs'], max_epochs=config['nepochs'], warmup_start_lr=config['warmup_start_lr'])

    # Train loop


    # Validation loop

    # Save logs and other information needed into the experiment folder
    
    return


if __name__ == "__main__":
    args = argument_parser()
    config = load_config(args)
    main(config)
    


