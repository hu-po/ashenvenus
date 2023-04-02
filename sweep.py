import uuid
import os

import pprint
import numpy as np
from hyperopt import fmin, hp, tpe
from train import train_loop

def objective(hparams) -> float:

    # Print hyperparam dict with logging
    print(f"\n\nHyperparams:\n\n{pprint.pformat(hparams)}\n\n")

    # Add UUID to run name for ultimate uniqueness
    run_name: str = str(uuid.uuid4())[:8] + '_'
    for key, value in hparams.items():
        # Choose name of run based on hparams
        if key in [
            'model',
            'freeze_backbone',
            'use_gelu',
            'curriculum',
            'optimizer',
            'lr_scheduling_gamma',
            'image_augs',
            # 'patch_size_x',
            # 'patch_size_y',
            'resize_ratio',
            # 'num_epochs',
            # 'batch_size',
            'lr',
            'max_samples_per_dataset',
        ]:
            run_name += f'{key}_{str(value)}_'

    # Create directory based on run_name
    output_dir = os.path.join(hparams['output_dir'], run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save hyperparams to file
    with open(os.path.join(output_dir, 'hparams.txt'), 'w') as f:
        f.write(pprint.pformat(hparams))
    
    try:
        # Train and evaluate a TFLite model
        score: float = train_loop(
            # Directories and datasets
            output_dir=output_dir,
            train_dir=hparams['train_dir'],
            eval_dir=hparams['eval_dir'],
            curriculum=hparams['curriculum'],
            image_augs=hparams['image_augs'],
            resize_ratio=hparams['resize_ratio'],
            num_workers=hparams['num_workers'],
            max_samples_per_dataset=hparams['max_samples_per_dataset'],
            # Model and training
            model=hparams['model'],
            freeze_backbone=hparams['freeze_backbone'],
            optimizer=hparams['optimizer'],
            lr_scheduling_gamma=hparams['lr_scheduling_gamma'],
            use_gelu=hparams['use_gelu'],
            slice_depth=hparams['slice_depth'],
            patch_size_x=hparams['patch_size_x'],
            patch_size_y=hparams['patch_size_y'],
            batch_size=hparams['batch_size'],
            lr=hparams['lr'],
            num_epochs=hparams['num_epochs'],
            save_pred_img=True,
            save_submit_csv=False,
            write_logs = True,
            save_model=True,
            max_time_hours = 8,
        )
    except Exception as e:
        print(f"\n\n Model Training FAILED with \n{e}\n\n")
        score = 0
    # Maximize score is minimize negative score
    return -score

# Define the search space
search_space = {
    # 'train_dir' : "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data\\train",
    # 'eval_dir' : "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data\\test",
    # 'output_dir' : "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\output",
    'output_dir': 'output',
    'train_dir': 'data/train',
    'eval_dir': 'data/test',
    'curriculum': hp.choice('curriculum', [
        # '1',
        # '2',
        # '3',
        '123',
        '321',
        '13',
        '32',
    ]),
    'model': hp.choice('model', [
        'simplenet',
        'convnext_tiny', # Good
        # 'swin_t',
        'resnext50_32x4d', # Potentially also good
        # 'vit_b_32',
    ]),
    'freeze_backbone': hp.choice('freeze_backbone', [
        True,
        False,
    ]),
    'use_gelu' : hp.choice('use_gelu', [
        True,
        False,
    ]),
    'image_augs': hp.choice('image_augs', [
        True,
        False,
    ]),
    'optimizer': hp.choice('optimizer', [
        'adam',
        # 'sgd', # Garbo
    ]),
    'lr_scheduling_gamma': hp.choice('lr_scheduling_gamma', [
        # 0.1, # Garbo
        0.9,
        0.98,
        None,
    ]),
    'slice_depth': 65,
    'num_workers': 0,
    'batch_size': hp.choice('batch_size', [32]),
    'lr': hp.loguniform('lr',  np.log(0.00000001), np.log(0.001)),
    'num_epochs': hp.choice('num_epochs', [8, 16]),
    'patch_size_x': hp.choice('patch_size_x', [64]),
    'patch_size_y': hp.choice('patch_size_y', [64]),
    'resize_ratio': hp.choice('resize_ratio', [0.08]),
    'max_samples_per_dataset': hp.choice('max_samples_per_dataset', [60000, 1000]),
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
