import argparse
import os
import pprint
import shutil
import uuid

import numpy as np
import yaml
from hyperopt import fmin, hp, tpe
from tensorboardX import SummaryWriter

from src import train_valid

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=1)

if os.name == 'nt':
    print("Windows Computer Detected")
    ROOT_DIR =  "C:\\Users\\ook\\Documents\\dev"
    DATA_DIR = os.path.join(ROOT_DIR, "ashenvenus\\data\\split")
    MODEL_DIR = os.path.join(ROOT_DIR, "ashenvenus\\models")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "ashenvenus\\output")
else:
    if os.path.isdir("/home/tren"):
        print("Linux Computer 1 Detected")
        ROOT_DIR = "/home/tren/dev/"
    elif os.path.isdir("/home/oop"):
        print("Linux Computer 2 Detected")
        ROOT_DIR = "/home/oop/dev/"
        DATA_DIR = os.path.join(ROOT_DIR, "ashenvenus/data/split")
        MODEL_DIR = os.path.join(ROOT_DIR, "ashenvenus/models")
        OUTPUT_DIR = os.path.join(ROOT_DIR, "ashenvenus/output")

# Define the search space
HYPERPARAMS = {
    'train_dir_name' : 'train',
    'valid_dir_name' : 'valid',
    # Model
    'model_str': hp.choice('model_str', [
        'vit_b|sam_vit_b_01ec64.pth',
        # 'vit_h|sam_vit_h_4b8939.pth',
        # 'vit_l|sam_vit_l_0b3195.pth',
    ]),

    # Dataset
    'curriculum': hp.choice('curriculum', [
        '1', # Depth of 1 - 40/45
        # '2', # Depth of 1 - 53/58
        '3', # Depth of 1 - 48/53
        '123',
    ]),
    'num_samples_train': hp.choice('num_samples_train', [
        # 2,
        2000,
    ]),
    'num_samples_valid': hp.choice('num_samples_valid', [
        # 2,
        10,
    ]),
    'resize': hp.choice('resize', [
        1.0, # Universal Harmonics
        # 0.3,
        # 0.3,
    ]),
    'crop_size_str': hp.choice('crop_size_str', [
        '256.256', # Universal Harmonics
        # '128.128',
        # '68.68',
    ]),
    'max_depth': hp.choice('max_depth', [
        42, # Universal Harmonics
    ]),
    'lr_sched': hp.choice('lr_sched', [
        'cosine',
        'gamma',
        'flat',
    ]),
    # Training
    'batch_size' : 2,
    'num_epochs': hp.choice('num_epochs', [
        4,
        8,
    ]),
    'warmup_epochs': hp.choice('warmup_epochs', [
        0,
        1,
    ]),
    'lr': hp.loguniform('lr',np.log(0.0001), np.log(0.01)),
    'wd': hp.choice('wd', [
        1e-4,
        1e-3,
        0,
    ]),
    'seed': 0,
}

def sweep_episode(hparams) -> float:

    # Print hyperparam dict with logging
    print(f"\n\n Starting EPISODE \n\n")
    print(f"\n\nHyperparams:\n\n{pprint.pformat(hparams)}\n\n")

    # Create output directory based on run_name
    run_name: str = str(uuid.uuid4())[:8]
    output_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Train and Validation directories
    train_dir = os.path.join(DATA_DIR, hparams['train_dir_name'])
    valid_dir = os.path.join(DATA_DIR, hparams['valid_dir_name'])

    # Save hyperparams to file with YAML
    with open(os.path.join(output_dir, 'hparams.yaml'), 'w') as f:
        yaml.dump(hparams, f)

    # HACK: Convert Hyperparam strings to correct format
    hparams['crop_size'] = [int(x) for x in hparams['crop_size_str'].split('.')]
    model, weights_filepath = hparams['model_str'].split('|')
    weights_filepath = os.path.join(MODEL_DIR, weights_filepath)

    try:
        writer = SummaryWriter(log_dir=output_dir)
        # Train and evaluate a TFLite model
        score_dict = train_valid(
            run_name =run_name,
            output_dir = output_dir,
            train_dir = train_dir,
            valid_dir = valid_dir,
            model=model,
            weights_filepath=weights_filepath,
            writer=writer,
            **hparams,
        )
        # Only works without tb open
        # writer.add_hparams(hparams, score_dict)
        writer.close()
        # Score is average of all scores
        score = sum(score_dict.values()) / len(score_dict)
    except Exception as e:
        print(f"\n\n (ERROR) EPISODE FAILED (ERROR) \n\n")
        print(f"Potentially Bad Hyperparams:\n\n{pprint.pformat(hparams)}\n\n")
        raise e
        # print(e)
        # score = 0
    # Maximize score is minimize negative score
    return -score

if __name__ == "__main__":

    # Clean output dir    
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    args = parser.parse_args()
    HYPERPARAMS['seed'] = args.seed
    HYPERPARAMS['batch_size'] = args.batch_size
    best = fmin(
        sweep_episode,
        space=HYPERPARAMS,
        algo=tpe.suggest,
        max_evals=100,
        rstate=np.random.Generator(np.random.PCG64(args.seed)),
    )