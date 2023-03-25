import logging
import os

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        log.info("Using GPU")
        return torch.device("cuda")
    else:
        log.info("Using CPU")
        return torch.device("cpu")

# Fast run length encoding, from https://www.kaggle.com/code/hackerpoet/even-faster-run-length-encoder/script


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


def write_to_submission(
    test_dir: str = "data/test",
    image_label_filename: str = "mask.png",
    submission_filepath: str = "submission.csv",
):
    assert os.path.exists(test_dir), f"File {test_dir} does not exist"
    # Create submission file if it does not exist
    if not os.path.exists(submission_filepath):
        with open(submission_filepath, 'w') as f:
            # Write header
            f.write("Id,Predicted")
    # Walk through the test directory
    for subtest_name in os.listdir(test_dir):
        log.info(f"Writing submission for {subtest_name}")
        # Name of sub-directory inside test dir
        subtest_filepath = os.path.join(test_dir, subtest_name)
        # Get mask image path inside directory
        image_label_filepath = os.path.join(
            subtest_filepath, image_label_filename)
        assert os.path.exists(
            image_label_filepath), f"File {image_label_filepath} does not exist"
        inklabels = np.array(Image.open(image_label_filepath), dtype=np.uint8)
        starts_ix, lengths = rle(inklabels)
        inklabels_rle = " ".join(map(str, sum(zip(starts_ix, lengths), ())))
        with open(submission_filepath, 'w') as f:
            f.write(f"{subtest_name},{inklabels_rle}")
