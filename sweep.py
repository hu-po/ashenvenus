import logging
import uuid

import pprint
import numpy as np
from hyperopt import fmin, hp, tpe
from train import train_valid_loop
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def objective(hparams) -> float:

    # Print hyperparam dict with logging
    log.info(f"\n\nHyperparams:\n\n{pprint.pformat(hparams)}\n\n")

    run_name: str = 'run'
    for key, value in hparams.items():
        # Choose name of run based on hparams
        if key in [
            'model',
            'curriculum',
            'optimizer',
            'image_augs',
            'patch_size_x',
            'patch_size_y',
            'resize_ratio',
            'num_epochs',
            'batch_size',
            'lr',
            'max_samples_per_dataset',
        ]:
            run_name += f'{key}_{str(value)}_'

    # Add UUID to run name for ultimate uniqueness
    run_name += str(uuid.uuid4())[:8]

    # Train and evaluate a TFLite model
    loss: float = train_valid_loop(
        train_dir=hparams['train_dir'],
        model=hparams['model'],
        optimizer=hparams['optimizer'],
        curriculum=hparams['curriculum'],
        image_augs=hparams['image_augs'],
        output_dir="output/train/",
        run_name=run_name,
        slice_depth=hparams['slice_depth'],
        patch_size_x=hparams['patch_size_x'],
        patch_size_y=hparams['patch_size_y'],
        resize_ratio=hparams['resize_ratio'],
        batch_size=hparams['batch_size'],
        lr=hparams['lr'],
        num_epochs=hparams['num_epochs'],
        num_workers=hparams['num_workers'],
        max_samples_per_dataset=hparams['max_samples_per_dataset'],
    )
    return loss


if __name__ == '__main__':
    log.setLevel(logging.DEBUG)
    args = parser.parse_args()

    # Define the search space
    search_space = {
        'train_dir': 'data/train',
        'curriculum': hp.choice('curriculum', [
            '1',
            '2',
            '3',
            '123',
            '321',
        ]),
        'model': hp.choice('model', [
            'simplenet',
            'simplenet_norm'
        ]),
        'image_augs': hp.choice('image_augs', [True, False]),
        'optimizer': hp.choice('optimizer', [
            'adam',
            'sgd'
        ]),
        'slice_depth': 65,
        'num_workers': 1,
        'batch_size': hp.choice('batch_size', [64]),
        'lr': hp.loguniform('lr',  np.log(0.00001), np.log(0.001)),
        'num_epochs': hp.choice('num_epochs', [6]),
        'patch_size_x': hp.choice('patch_size_x', [128]),
        'patch_size_y': hp.choice('patch_size_y', [64]),
        'resize_ratio': hp.choice('resize_ratio', [0.1]),
        'max_samples_per_dataset': hp.choice('max_samples_per_dataset', [1000, 10000]),
    }
    if args.seed == 420:
        print('TEST MODE')
        search_space['max_samples_per_dataset'] = 100
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
