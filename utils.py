import torch
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        log.info("Using GPU")
        return torch.device("cuda")
    else:
        log.info("Using CPU")
        return torch.device("cpu")

def rle(output, threshold=0.4):
    flat_img = np.where(output.flatten().cpu() >
                        threshold, 1, 0).astype(np.uint8)
    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    return " ".join(map(str, sum(zip(starts_ix, lengths), ())))


def make_submission():
    submission = defaultdict(list)
    for fragment_id, fragment_name in enumerate(test_fragments):
        submission["Id"].append(fragment_name.name)
        submission["Predicted"].append(rle(pred_images[fragment_id]))

    pd.DataFrame.from_dict(submission).to_csv(
        "/kaggle/working/submission.csv", index=False)
