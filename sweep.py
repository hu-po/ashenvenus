import uuid
import os

import pprint
import numpy as np
from hyperopt import fmin, hp, tpe
from train import train_loop
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)

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
            'curriculum',
            'optimizer',
            'lr_scheduling_gamma',
            # 'image_augs',
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
        loss: float = train_loop(
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
        loss = 10000.0
    return loss


if __name__ == '__main__':
    args = parser.parse_args()

    # Define the search space
    search_space = {
        'output_dir': 'output',
        'train_dir': 'data/train',
        'eval_dir': 'data/test',
        'curriculum': hp.choice('curriculum', [
            '1',
            '2',
            '3',
            '123',
            '321',
            '13',
        ]),
        'model': hp.choice('model', [
            'simplenet',
            'simplenet_norm',
            'convnext_tiny',
            'swin_t',
            'resnext50_32x4d',
            'vit_b_32',
        ]),
        'freeze_backbone': hp.choice('freeze_backbone', [
            True,
            False,
        ]),
        'image_augs': hp.choice('image_augs', [
            True,
            False,
        ]),
        'optimizer': hp.choice('optimizer', [
            'adam',
            # 'sgd', # Trains slower and no good
        ]),
        'lr_scheduling_gamma': hp.choice('lr_scheduling_gamma', [
            0.1,
            0.9,
            # None,
        ]),
        'slice_depth': 65,
        'num_workers': 1,
        'batch_size': hp.choice('batch_size', [args.batch_size]),
        'lr': hp.loguniform('lr',  np.log(0.000001), np.log(0.01)),
        'num_epochs': hp.choice('num_epochs', [6]),
        'patch_size_x': hp.choice('patch_size_x', [224]),
        'patch_size_y': hp.choice('patch_size_y', [224]),
        'resize_ratio': hp.choice('resize_ratio', [0.25]),
        'max_samples_per_dataset': hp.choice('max_samples_per_dataset', [80000]),
    }
    if args.seed == 420:
        print('TEST MODE')
        search_space['max_samples_per_dataset'] = 64
        search_space['num_epochs'] = 2
        search_space['slice_depth'] = 2
        search_space['resize_ratio'] = 0.05

    # Run the optimization
    best = fmin(
        objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=100,
        rstate=np.random.default_rng(args.seed),
    )
