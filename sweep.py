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
    _acc: float = train_valid_loop(
        train_dir="data/train/1",
        output_dir="output/train",
        run_name="debug",
        slice_depth=hparams['slice_depth'],
        patch_size_x=hparams['patch_size_x'],
        patch_size_y=hparams['patch_size_y'],
        resize_ratio=hparams['resize_ratio'],
        batch_size=hparams['batch_size'],
        lr=hparams['lr'],
        num_epochs=hparams['num_epochs'],
        num_workers=hparams['num_workers'],
    )
    return _acc


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)
    # Set logging for matplotlib module to info
    logging.getLogger('matplotlib').setLevel(logging.INFO)

    # Define the search space
    search_space = {
        'slice_depth': 65,
        'num_workers': 16,
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'lr': hp.loguniform('lr',  np.log(0.00001), np.log(0.01)),
        'num_epochs': hp.choice('num_epochs', [2, 6, 10]),
        'patch_size_x': hp.choice('patch_size_x', [32, 64, 128, 512]),
        'patch_size_y': hp.choice('patch_size_y', [32, 64, 128, 512]),
    }

    # Run the optimization
    best = fmin(
        objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        rstate=np.random.default_rng(0),
    )
