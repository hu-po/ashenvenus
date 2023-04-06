import argparse
import shutil

import numpy as np
from hyperopt import fmin, hp, tpe

from src import sweep_episode

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resize',  type=float, default=None)

# Define the search space
search_space = {
    'output_dir': 'output',
    'train_dir': 'data/split_train',
    'valid_dir': 'data/split_valid',
    'curriculum': hp.choice('curriculum', [
        # All 3 performs better, order doesn't seem to matter
        # '1',
        # '2',
        # '3',
        # '13',
        # '32',
        '123',
        # '321',
    ]),
    'model': hp.choice('model', [
        # 'convnext_tiny',
        'convnext_small',
        'convnext_base',
        # 'convnext_large',
        'resnext50_32x4d',
        # 'resnext101_32x8d',
        # 'resnext101_64x4d',
        # ViTs are picky about input size
        # 'vit_b_32', 
        # 'vit_l_32',
        # 'vit_h_14',
    ]),
    'freeze': hp.choice('freeze', [
        # Doesn't seem to matter much, which is odd
        True,
        # False,
    ]),
    'image_augs': hp.choice('image_augs', [
        # Sweep 060423 - Mostly helps a little
        # Sweep 040423 - Hurts a little more than helps, but doesn't matter much
        True,
        # False,
    ]),
    'optimizer': hp.choice('optimizer', [
        'adam',
        # 'sgd', # Garbo
    ]),
    'weight_decay': hp.choice('weight_decay', [
        # Hard to tell, good runs exist with and without
        1e-4,
        1e-3,
    ]),
    'lr_gamma': hp.choice('lr_gamma', [
        # Doesn't seem to matter much between 0.9 and None (1.0)
        # 0.1, # Garbo
        # 0.9,
        0.98,
        # None,
    ]),
    'num_workers': 0,
    'resize_ratio': hp.choice('resize_ratio', [
        # Below 0.1 it can't learn
        # Above 0.3 it takes hours to perform eval
        0.2,
    ]),
    'interpolation': hp.choice('interpolation', [
        # Bicubic is best, but nearest is faster
        # 'nearest',
        'bilinear',
        'bicubic',
    ]),
    'input_size': hp.choice('input_size', [
        # Smaller resolutions actually work quite well
        # '224.224.65',
        '32.32.65',
        '64.64.65',
        '128.64.65',
    ]),
    # Careful with small learning rates, less than 1e-5 is too small
    'lr': hp.loguniform('lr',np.log(0.00001), np.log(0.01)),
    'num_epochs': hp.choice('num_epochs', [8]),
    'num_samples_train': hp.choice('num_samples_train', [
        # Larger is better, strongest predictor of score
        120000,
        # 120000,
        # 60000,
        # 4000,
        # 2000,
    ]),
    'num_samples_valid': hp.choice('num_samples_valid', [
        # Larger is more thorough, but takes more time
        4000
    ]),
    'max_time_hours': hp.choice('max_time_hours', [
        # 4,
        9,
    ]),
    'threshold': hp.choice('threshold', [0.5]),
    'postproc_kernel': hp.choice('postproc_kernel', [3]),
}

if __name__ == '__main__':
    args = parser.parse_args()
    if args.seed == 0:
        print('\n\n Running in TEST mode \n\n')
        search_space['interpolation'] = 'nearest'
        search_space['curriculum'] = '1'
        search_space['num_samples_train'] = 64
        search_space['num_samples_valid'] = 64
        search_space['num_epochs'] = 2
    search_space['batch_size'] = args.batch_size
    if args.resize:
        search_space['resize_ratio'] = args.resize

    # Clean output dir    
    shutil.rmtree(search_space['output_dir'], ignore_errors=True)

    best = fmin(
        sweep_episode,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
        rstate=np.random.Generator(np.random.PCG64(args.seed)),
    )