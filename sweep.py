import os
import pprint
import uuid

import numpy as np
import yaml
from hyperopt import fmin, hp, tpe

from train import train_loop




# Define the search space
search_space = {
    # 'train_dir' : "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data\\train",
    # 'eval_dir' : "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data\\test",
    # 'output_dir' : "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\output",
    'output_dir': 'output',
    'train_dir': 'data/train',
    'eval_dir': 'data/test',
    'hparams_filename': 'hparams.yaml',
    'curriculum': hp.choice('curriculum', [
        # All 3 performs better, order doesn't seem to matter
        # '1',
        # '2',
        # '3',
        # '13',
        # '32',
        '123',
        '321',
    ]),
    'model': hp.choice('model', [
        # Doesn't seem to matter much, all models perform similarly, even simplenet
        'simplenet',
        'convnext_tiny', # Good
        # 'swin_t',
        'resnext50_32x4d', # Potentially also good
        # 'vit_b_32',
    ]),
    'freeze_backbone': hp.choice('freeze_backbone', [
        # Doesn't seem to matter much, which is odd
        True,
        False,
    ]),
    'use_gelu' : hp.choice('use_gelu', [
        # Doesn't seem to matter much, maybe slight gain
        True,
        # False,
    ]),
    'image_augs': hp.choice('image_augs', [
        # Hurts a little more than helps, but doesn't matter much
        # True,
        False,
    ]),
    'optimizer': hp.choice('optimizer', [
        'adam',
        # 'sgd', # Garbo
    ]),
    'lr_scheduling_gamma': hp.choice('lr_scheduling_gamma', [
        # Doesn't seem to matter much between 0.9 and None (1.0)
        # 0.1, # Garbo
        # 0.9,
        0.98,
        # None,
    ]),
    'slice_depth': 65,
    'num_workers': 0,
    'batch_size': hp.choice('batch_size', [32]),
    'lr': hp.loguniform('lr',  np.log(0.0000001), np.log(0.001)),
    'num_epochs': hp.choice('num_epochs', [8, 16]),
    'patch_size_x': hp.choice('patch_size_x', [64]),
    'patch_size_y': hp.choice('patch_size_y', [64]),
    'resize_ratio': hp.choice('resize_ratio', [0.08]),
    'max_samples_per_dataset': hp.choice('max_samples_per_dataset', [
        # Larger is better, strongest predictor of score
        60000,
        1000,
    ]),
}

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
if args.seed == 420:
    print('TEST MODE')
    search_space['max_samples_per_dataset'] = 64
    search_space['num_epochs'] = 2
    search_space['slice_depth'] = 2
    search_space['resize_ratio'] = 0.05
search_space['batch_size'] = args.batch_size

# Run the optimization
best = fmin(
    objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=100,
    rstate=np.random.default_rng(args.seed),
)
