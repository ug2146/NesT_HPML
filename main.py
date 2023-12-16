# Baseline Nested Hierarchical Transformer training

import argparse
import logging
import os
import yaml

import torch
import time


from pprint import pprint
from trainer import Trainer
from utils.logging import config_logging


def argument_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/default.yaml', help='YAML configuration file')
    add_arg('--name', help='name of the experiment. Also used as output directory')
    add_arg('--model', help='model to be used')
    add_arg('--dataset', help='dataset to be used')
    add_arg('--train-batch-size', '-train-bs', help='training batch size')
    add_arg('--test-batch-size', '-test-bs', help='test batch size')
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

    print("Current configuration: ")
    pprint(config)

    # Create output/experiment folder and save config file
    output_dir = os.path.join("experiments", config['name'])
    os.makedirs(output_dir, exist_ok = True)
    with open(os.path.join(output_dir, "config.yaml"), 'w') as outfile:
        yaml.dump(config, outfile, sort_keys=False, default_flow_style=False)
    
    # Support for only single-gpu training
    assert config['ngpus'] == 1, "Sorry! Only single-gpu training is supported!"

    trainer = Trainer(config, output_dir)

    if torch.cuda.is_available():   torch.cuda.synchronize()
    exp_start = time.perf_counter()
    trainer.train()
    if torch.cuda.is_available():   torch.cuda.synchronize()
    exp_end = time.perf_counter()
    logging.info(f"Average run time per epoch: {(exp_end - exp_start)/config['nepochs']:.3f} s")
    return

if __name__ == "__main__":
    args = argument_parser()
    config = load_config(args)
    main(config)
    


