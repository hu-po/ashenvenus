import logging
import uuid

import numpy as np
from hyperopt import fmin, hp, tpe
from train import train_valid_loop


def objective(hparams) -> float:

    run_name: str = 'run'
    for key, value in hparams.items():
        # Choose name of run based on hparams
        if key in [
            'patch_size_x',
            'patch_size_y',
            'resize_ratio',
            'num_epochs',
            'batch_size',
            'lr',
        ]:
            run_name += f'_{str(value)}'

    # Add UUID to run name for ultimate uniqueness
    run_name += str(uuid.uuid4())[:8]

    # Train and evaluate a TFLite model
    loss: float = train_valid_loop(
        train_dir=hparams['train_dir'],
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
    )
    return loss


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    # Set logging for matplotlib module to info
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    # Define the search space
    search_space = {
        'train_dir': hp.choice('train_dir', ['data/train/1', 'data/train/2', 'data/train/3']),
        'slice_depth': 65,
        'num_workers': 1,
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'lr': hp.loguniform('lr',  np.log(0.00001), np.log(0.01)),
        'num_epochs': hp.choice('num_epochs', [2, 6, 10]),
        'patch_size_x': hp.choice('patch_size_x', [32, 64, 128, 512]),
        'patch_size_y': hp.choice('patch_size_y', [32, 64, 128, 512]),
        'resize_ratio': hp.choice('resize_ratio', [0.1, 0.25, 0.5]),
    }

    # Run the optimization
    best = fmin(
        objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        rstate=np.random.default_rng(0),
    )
