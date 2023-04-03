import argparse
import shutil

import numpy as np
from hyperopt import fmin, hp, tpe

from src import sweep_episode

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=0)

# Define the search space
search_space = {
    'output_dir': 'output',
    'train_dir': 'data/train',
    'eval_dir': 'data/test',
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
        'convnext_tiny',
        # 'convnext_small',
        # 'convnext_base',
        # 'convnext_large',
        'resnext50_32x4d',
        # 'resnext101_32x8d',
        # 'resnext101_64x4d',
    ]),
    'freeze': hp.choice('freeze', [
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
    'lr_gamma': hp.choice('lr_gamma', [
        # Doesn't seem to matter much between 0.9 and None (1.0)
        # 0.1, # Garbo
        # 0.9,
        0.98,
        # None,
    ]),
    'slice_depth': 65,
    'num_workers': 0,
    'resize_ratio': hp.choice('resize_ratio', [0.08]),
    'patch_size_x': hp.choice('patch_size_x', [64]),
    'patch_size_y': hp.choice('patch_size_y', [64]),
    'batch_size': hp.choice('batch_size', [32]),
    'lr': hp.loguniform('lr',  np.log(0.0000001), np.log(0.001)),
    'num_epochs': hp.choice('num_epochs', [8, 16]),
    'num_samples': hp.choice('num_samples', [
        # Larger is better, strongest predictor of score
        60000,
        1000,
    ]),
    'max_time_hours': hp.choice('max_time_hours', [2, 8]),
    'threshold': hp.choice('threshold', [0.5]),
    'postproc_kernel': hp.choice('postproc_kernel', [3, 6]),
}

if __name__ == '__main__':
    args = parser.parse_args()
    if args.seed == 420:
        print('TEST MODE')
        search_space['curriculum'] = '1'
        search_space['num_samples'] = 64
        search_space['num_epochs'] = 2
        search_space['slice_depth'] = 2
        search_space['resize_ratio'] = 0.05
    search_space['batch_size'] = args.batch_size

    # Clean output dir    
    shutil.rmtree(search_space['output_dir'], ignore_errors=True)

    best = fmin(
        sweep_episode,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
        rstate=np.random.Generator(np.random.PCG64(args.seed)),
    )