import gc
import subprocess

import numpy as np
import torch


def get_device():
    if torch.cuda.is_available():
        print("Using GPU")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


def rle(img, threshold=0.5):
    # TODO: Histogram of image to see where threshold should be
    flat_img = img.flatten()
    flat_img = np.where(flat_img > threshold, 1, 0).astype(np.uint8)
    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    return starts_ix, lengths


def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free',
                             '--format=csv,nounits,noheader'], stdout=subprocess.PIPE, text=True)
    gpu_memory = [tuple(map(int, line.split(',')))
                  for line in result.stdout.strip().split('\n')]
    for i, (used, free) in enumerate(gpu_memory):
        print(f"GPU {i}: Memory Used: {used} MiB | Memory Available: {free} MiB")


def clear_gpu_memory():
    if torch.cuda.is_available():
        print('Clearing GPU memory')
        torch.cuda.empty_cache()
        gc.collect()
