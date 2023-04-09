import os
import numpy as np
import shutil
import pprint
import uuid
import yaml
from tensorboardX import SummaryWriter
from hyperopt import fmin, hp, tpe
from src import train_valid

if os.name == 'nt':
    print("Windows Computer Detected")
    ROOT_DIR = "C:\\Users\\ook\\Documents\\dev\\"
    DATA_DIR = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data"
    MODEL_DIR = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\models"
    OUTPUT_DIR = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\output"
else:
    print("Linux Computer Detected")
    ROOT_DIR = "/home/tren/dev/"
    DATA_DIR = "/home/tren/dev/ashenvenus/data"
    MODEL_DIR = "/home/tren/dev/ashenvenus/models"
    OUTPUT_DIR = "/home/tren/dev/ashenvenus/output"

# Define the search space
HYPERPARAMS = {
    'train_dir_name' : 'split_train',
    'valid_dir_name' : 'split_valid',
    # Model
    'model_str': hp.choice('model_str', [
        'vit_b|sam_vit_b_01ec64.pth',
        # 'vit_h|sam_vit_h_4b8939.pth',
        # 'vit_l|sam_vit_l_0b3195.pth',
    ]),

    # Dataset
    'curriculum': hp.choice('curriculum', [
        '1', # Depth of 1 - 40/45
        '2', # Depth of 1 - 53/58
        '3', # Depth of 1 - 48/53
        # '123',
    ]),
    'num_samples_train': hp.choice('num_samples_train', [
        8,
    ]),
    'num_samples_valid': hp.choice('num_samples_valid', [
        8,
    ]),
    'crop_size_str': hp.choice('crop_size_str', [
        '3.1024.1024', # HACK: This cannot be changed for pretrained models
    ]),
    'label_size_str': hp.choice('label_size_str', [
        '256.256', # HACK: This cannot be changed for pretrained models
    ]),

    # Training
    'batch_size' : 1,
    'num_epochs': hp.choice('num_epochs', [8]),
    'lr': hp.loguniform('lr',np.log(0.00001), np.log(0.01)),
    'wd': hp.choice('wd', [
        1e-4,
        1e-3,
    ]),
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
    hparams['label_size'] = [int(x) for x in hparams['label_size_str'].split('.')]
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
        score = 0
    # Maximize score is minimize negative score
    return -score

if __name__ == "__main__":

    # Clean output dir    
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    best = fmin(
        sweep_episode,
        space=HYPERPARAMS,
        algo=tpe.suggest,
        max_evals=100,
        rstate=np.random.Generator(np.random.PCG64(42)),
    )