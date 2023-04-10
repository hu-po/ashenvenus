import csv
import gc
import os
from typing import Dict, Tuple
from io import StringIO
import PIL.Image as Image
import yaml
import pprint

import cv2
import numpy as np
import torch
import torch.nn as nn
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
)
from segment_anything.modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
)
from segment_anything.utils.amg import (
    MaskData,
    build_point_grid,
    batched_mask_to_box,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def get_device(device: str = None):
    if device == None or device == "gpu":
        if torch.cuda.is_available():
            print("Using GPU")
            print("Clearing GPU memory")
            torch.cuda.empty_cache()
            gc.collect()
            return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


def image_to_rle_fast(img):
    img[0] = 0
    img[-1] = 0
    runs = np.where(img[1:] != img[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    f = StringIO()
    np.savetxt(f, runs.reshape(1, -1), delimiter=" ", fmt="%d")
    return f.getvalue().strip()


def save_rle_as_image(rle_csv_path, output_dir, subtest_name, image_shape):
    with open(rle_csv_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            _subtest_name, rle_data = row
            if _subtest_name != subtest_name:
                continue
            rle_pairs = list(map(int, rle_data.split()))

            # Decode RLE data
            img = np.zeros(image_shape[0] * image_shape[1], dtype=np.uint8)
            for i in range(0, len(rle_pairs), 2):
                start = rle_pairs[i] - 1
                end = start + rle_pairs[i + 1]
                img[start:end] = 1

            # Reshape decoded image data to original shape
            img = img.reshape(image_shape)
            img = Image.fromarray(img * 255).convert("1")
            _image_filepath = os.path.join(output_dir,
                                           f"pred_{subtest_name}_rle.png")
            img.save(_image_filepath)


def dice_score(preds, label, beta=0.5, epsilon=1e-6):
    preds = torch.sigmoid(preds)
    preds = preds.flatten()
    label = label.flatten()
    tp = preds[label == 1].sum()
    fp = preds[label == 0].sum()
    fn = label.sum() - tp
    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)
    _score = (1 + beta * beta) * (p * r) / (beta * beta * p + r + epsilon)
    return _score


class FragmentDataset(Dataset):
    def __init__(
        self,
        # Directory containing the dataset
        data_dir: str,
        # Number of random crops to take from fragment volume
        dataset_size: int = 2,
        # Number of points to sample per crop
        points_per_crop: int = 4,
        # Filenames of the images we'll use
        image_mask_filename="mask.png",
        image_labels_filename="inklabels.png",
        slices_dir_filename="surface_volume",
        ir_image_filename="ir.png",
        # Expected slices per fragment
        crop_size: Tuple[int] = (3, 68, 68),
        label_size: Tuple[int] = (256, 256),
        # Depth in scan is a Clipped Normal distribution
        min_depth: int = 0,
        max_depth: int = 65,
        avg_depth: float = 27.0,
        std_depth: float = 10.0,
        # Training vs Testing mode
        train: bool = True,
        # Device to use
        device: str = "cuda",
    ):
        print(f"Making Dataset from {data_dir}")
        self.dataset_size = dataset_size
        self.points_per_crop = points_per_crop
        self.train = train
        self.device = device

        # Open Mask image
        _image_mask_filepath = os.path.join(data_dir, image_mask_filename)
        self.mask = np.array(
            cv2.imread(_image_mask_filepath,
                       cv2.IMREAD_GRAYSCALE)).astype(np.uint8)
        # Image dimmensions (depth, height, width)
        self.original_size = self.mask.shape
        self.crop_size = crop_size
        self.label_size = label_size
        # Open Label image
        if self.train:
            _image_labels_filepath = os.path.join(data_dir,
                                                  image_labels_filename)
            self.labels = np.array(
                cv2.imread(_image_labels_filepath,
                           cv2.IMREAD_GRAYSCALE)).astype(np.uint8)
        # Slices
        self.slice_dir = os.path.join(data_dir, slices_dir_filename)
        # Sample random crops within the image
        self.indices = np.zeros((dataset_size, 2, 3), dtype=np.int64)
        for i in range(dataset_size):
            # Select a random starting point for the subvolume
            start_depth = int(
                np.clip(np.random.normal(avg_depth, std_depth), min_depth,
                        max_depth))
            start_height = np.random.randint(
                0, self.original_size[0] - self.crop_size[1])
            start_width = np.random.randint(
                0, self.original_size[1] - self.crop_size[2])
            self.indices[i, 0, :] = [start_depth, start_height, start_width]
            # End point is start point + crop size
            self.indices[i, 1, :] = [
                start_depth + self.crop_size[0],
                start_height + self.crop_size[1],
                start_width + self.crop_size[2],
            ]
        # DEBUG: IR image
        _image_ir_filepath = os.path.join(data_dir, ir_image_filename)
        self.ir_image = np.array(cv2.imread(_image_ir_filepath)).astype(
            np.float32)
        self.ir_image = np.transpose(self.ir_image, (2, 0, 1))

        # Pixel stats for ir image, only for values inside mask
        self.pixel_mean = np.mean(self.ir_image[:, self.mask == 1])
        self.pixel_std = np.std(self.ir_image[:, self.mask == 1])

        # TODO: Pixel stats for points inside labels, outside labels for all slices
        # This might be better calculated beforehand and then loaded

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Start and End points for the crop in pixel space
        start = self.indices[idx, 0, :]
        end = self.indices[idx, 1, :]

        # Create a grid of points in the crop
        points = build_point_grid(self.points_per_crop)
        points = torch.from_numpy(points).to(device=self.device)

        # Get the label for each point
        point_labels = torch.zeros((self.points_per_crop**2), dtype=torch.long)
        for i, point in enumerate(points):
            point_labels[i] = self.labels[point[1], point[2]]
        point_labels = point_labels.to(device=self.device)

        # # Load the relevant slices and pack into image tensor
        # image = np.zeros(self.crop_size, dtype=np.float32)
        # for i, _depth in enumerate(range(start[0], end[0])):
        #     _slice_filepath = os.path.join(self.slice_dir, f"{_depth:02d}.tif")
        #     _slice = np.array(cv2.imread(_slice_filepath, cv2.IMREAD_GRAYSCALE)).astype(np.float32)
        #     image[i, :, :] = _slice[start[1]: end[1], start[2]: end[2]]
        # image = torch.from_numpy(image).to(device=self.device)

        # TODO: Take a 3D Volume and show the top, back, left, right view of volume
        # Bias towards a longer height than width
        # Try to get the entire depth
        # Find some tiling of the width and height such that you get 1024x1024
        # 65x128x3

        # DEBUG: Use IR image instead of surface volume as toy problem
        image = self.ir_image[:, start[1]:end[1], start[2]:end[2]]
        image = torch.from_numpy(image).to(device=self.device)

        # Normalize image
        image = (image - self.pixel_mean) / self.pixel_std

        if self.train:
            labels = self.labels[start[1]:end[1], start[2]:end[2]]
            low_res_labels = cv2.resize(
                labels.astype(np.uint8),
                self.label_size,
                interpolation=cv2.INTER_NEAREST,
            )
            low_res_labels = torch.from_numpy(low_res_labels).to(
                dtype=torch.float32)
            low_res_labels = low_res_labels.unsqueeze(0).clone().to(
                device=self.device)
            return image, points, point_labels, low_res_labels, labels
        else:
            return image


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def binary_iou(self, mask1, mask2):
        intersection = (mask1 * mask2).sum(dim=(-1, -2))
        union = torch.logical_or(mask1, mask2).sum(dim=(-1, -2))
        iou = intersection.float() / union.float()
        return iou

    def forward(self, logits, gt_masks, predicted_iou):
        # Calculate pixel-wise binary cross-entropy loss
        bce_loss = self.bce_with_logits_loss(logits, gt_masks.float())

        # Calculate predicted masks
        pred_masks = torch.sigmoid(logits) >= 0.5
        pred_masks = pred_masks.float()

        # Calculate actual IoU scores for each pair in the batch
        actual_iou = self.binary_iou(pred_masks, gt_masks)

        # Calculate the MSE loss between predicted and actual IoU scores
        iou_loss = self.mse_loss(predicted_iou, actual_iou)

        # Combine the two losses using a weighting factor alpha
        combined_loss = self.alpha * bce_loss + (1 - self.alpha) * iou_loss

        return combined_loss


def train_valid(
    run_name: str = "testytest",
    output_dir: str = None,
    train_dir: str = None,
    valid_dir: str = None,
    # Model
    model: str = "vit_b",
    weights_filepath: str = "path/to/model.pth",
    save_model: bool = True,
    # Training
    device: str = None,
    num_samples_train: int = 2,
    num_samples_valid: int = 2,
    num_epochs: int = 2,
    batch_size: int = 1,
    optimizer: str = "adam",
    lr: float = 1e-5,
    wd: float = 1e-4,
    alpha: float = 0.5,
    writer=None,
    log_images: bool = False,
    # Dataset
    curriculum: str = "1",
    crop_size: Tuple[int] = (3, 68, 68),
    label_size: Tuple[int] = (1024, 1024),
    points_per_crop: int = 8,
    avg_depth: float = 27.0,
    std_depth: float = 10.0,
    **kwargs,
):
    device = get_device(device)
    model = sam_model_registry[model](checkpoint=weights_filepath)
    # TODO: Which of these should be frozen?
    # for param in model.image_encoder.parameters():
    #         param.requires_grad = False
    # for param in model.prompt_encoder.parameters():
    #         param.requires_grad = False
    # for param in model.mask_decoder.parameters():
    #     param.requires_grad = False
    model.to(device=device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = CombinedLoss(alpha=alpha)
    # TODO: Learning rate scheduler
    # TODO: Learning rate warmup

    train_step = 0
    best_score_dict: Dict[str, float] = {}
    for epoch in range(num_epochs):
        print(f"\n\n --- Epoch {epoch+1} of {num_epochs} --- \n\n")
        for phase, data_dir, num_samples in [("Train", train_dir, num_samples_train), ("Valid", valid_dir, num_samples_valid)]:
            for _dataset_id in curriculum:
                _dataset_filepath = os.path.join(data_dir, _dataset_id)
                print(f"{phase} on {_dataset_filepath} ...")
                _score_name = f"Dice/{phase}/{_dataset_id}"
                if _score_name not in best_score_dict:
                    best_score_dict[_score_name] = 0
                _dataset = FragmentDataset(
                    data_dir=_dataset_filepath,
                    dataset_size=num_samples,
                    points_per_crop=points_per_crop,
                    crop_size=crop_size,
                    label_size=label_size,
                    avg_depth=avg_depth,
                    std_depth=std_depth,
                    train=True,
                    device=device,
                )
                _dataloader = DataLoader(
                    dataset=_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=True,
                )
                _loader = tqdm(_dataloader)
                score = 0
                for images, points, point_labels, low_res_labels, labels in _loader:
                    train_step += 1
                    if writer and log_images:
                        writer.add_images(f"input-image/{phase}/{_dataset_id}",
                                          images, train_step)
                        writer.add_images(f"input-label/{phase}/{_dataset_id}",
                                          labels * 255, train_step)
                        # Plot point coordinates into a blank image of size images
                        _point_coords = points.cpu().numpy()
                        _point_labels = point_labels.cpu().numpy()
                        _point_image = np.zeros(
                            (1, 3, labels.shape[2], labels.shape[3]),
                            dtype=np.uint8)
                        _point_image[
                            0,
                            2, :, :] = labels.cpu().numpy()[0, 0, :, :] * 255
                        point_width = 4
                        for i in range(_point_coords.shape[1]):
                            _height = _point_coords[0, i, 0]
                            _width = _point_coords[0, i, 1]
                            if _point_labels[0, i] == 0:
                                _point_image[0, 0, _height -
                                             point_width:_height + point_width,
                                             _width - point_width:_width +
                                             point_width] = 255
                            else:
                                _point_image[0, 1, _height -
                                             point_width:_height + point_width,
                                             _width - point_width:_width +
                                             point_width] = 255
                        writer.add_images(
                            f"input-points/{phase}/{_dataset_id}",
                            _point_image, train_step)
                        writer.flush()
                    image_embeddings = model.image_encoder(images)
                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=(points, point_labels),
                        # TODO: These boxes and labels might have to be per-point
                        boxes=batched_mask_to_box(labels),
                        masks=labels,
                    )
                    low_res_masks, iou_predictions = model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    if writer and log_images:
                        writer.add_images(
                            f"output.masks/{phase}/{_dataset_id}",
                            low_res_masks, train_step)
                    loss = loss_fn(low_res_masks, low_res_labels,
                                   iou_predictions)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    _loss_name = f"{loss_fn.__class__.__name__}/{phase}/{_dataset_id}"
                    _loader.set_postfix_str(f"{_loss_name}: {loss.item():.4f}")
                    if writer:
                        writer.add_scalar(f"{_loss_name}", loss.item(),
                                          train_step)

                    score += dice_score(low_res_masks, labels)
                score /= len(_dataloader)
                if writer:
                    writer.add_scalar(f"Dice/{phase}", score, train_step)
                # Overwrite best score if it is better
                if score > best_score_dict[_score_name]:
                    print(f"New best score! {score:.4f} ")
                    print(f"(was {best_score_dict[_score_name]:.4f})")
                    best_score_dict[_score_name] = score
                    if save_model:
                        _model_filepath = os.path.join(
                            output_dir,
                            f"model_{run_name}_best_{_dataset_id}.pth")
                        print(f"Saving model to {_model_filepath}")
                        torch.save(model.state_dict(), _model_filepath)
        # Flush writer every epoch
        writer.flush()
    writer.close()
    return best_score_dict


def eval(
    output_dir: str = None,
    eval_dir: str = None,
    save_pred_img: bool = True,
    save_submit_csv: bool = False,
    # Evaluation
    batch_size: int = 1,
    num_samples_eval: int = 100,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
    # Model
    model: str = "vit_b",
    weights_filepath: str = None,
    # Dataset
    crop_size: Tuple[int] = (3, 68, 68),
    label_size: Tuple[int] = (1024, 1024),
    points_per_crop: int = 8,
    avg_depth: float = 27.0,
    std_depth: float = 10.0,
    **kwargs,
):
    device = get_device(device)
    model = SamAutomaticMaskGenerator(
        model=sam_model_registry[model](checkpoint=weights_filepath),
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
    )
    model.eval()
    model.to(device)

    if save_submit_csv:
        submission_filepath = os.path.join(output_dir, "submission.csv")
        with open(submission_filepath, "w") as f:
            # Write header
            f.write("Id,Predicted\n")

    # Baseline is to use image mask to create guess submission
    for dataset_id in os.listdir(eval_dir):
        print(f"Evaluating on {dataset_id}")
        _dataset_filepath = os.path.join(eval_dir, dataset_id)
        _dataset = FragmentDataset(
            data_dir=_dataset_filepath,
            dataset_size=num_samples_eval,
            points_per_crop=points_per_crop,
            crop_size=crop_size,
            label_size=label_size,
            avg_depth=avg_depth,
            std_depth=std_depth,
            train=False,
            device=device,
        )
        _dataloader = DataLoader(
            dataset=_dataset,
            batch_size=batch_size,
            shuffle=False,
            # pin_memory=True,
        )
        mask_data: MaskData = None
        _loader = tqdm(_dataloader)
        for img in _loader:

            # Get masks from image
            _mask_data: MaskData = model.generate(img)
            """
                `segmentation` : the mask
                `area` : the area of the mask in pixels
                `bbox` : the boundary box of the mask in XYWH format
                `predicted_iou` : the model's own prediction for the quality of the mask
                `point_coords` : the sampled input point that generated this mask
                `stability_score` : an additional measure of mask quality
                `crop_box` : the crop of the image used to generate this mask in XYWH format
            """
            # Filter the masks using NMS

            # Group all the predicted masks together
            if mask_data is None:
                mask_data = _mask_data
            else:
                mask_data.cat(_mask_data)

        # Convert masks to single image

        if save_pred_img:
            print("Saving prediction image...")
            _image_filepath = os.path.join(output_dir,
                                           f"pred_{dataset_id}.png")
            img.save(_image_filepath)

        if save_submit_csv:
            print("Saving submission csv...")
            img = np.array(img).flatten()
            inklabels_rle_fast = image_to_rle_fast(img)
            with open(submission_filepath, "a") as f:
                f.write(f"{dataset_id},{inklabels_rle_fast}\n")


def eval_from_episode_dir(
    episode_dir: str = None,
    output_dir: str = None,
    weights_filename: str = "model.pth",
    hparams_filename: str = "hparams.yaml",
    **kwargs,
):
    # Get hyperparams from text file
    _hparams_filepath = os.path.join(episode_dir, hparams_filename)
    with open(_hparams_filepath, "r") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    _weights_filepath = os.path.join(episode_dir, weights_filename)
    # Merge kwargs with hparams, kwargs takes precedence
    hparams = {**hparams, **kwargs}
    print(f"Hyperparams:\n{pprint.pformat(hparams)}\n")
    eval(
        output_dir=output_dir,
        weights_filepath=_weights_filepath,
        **hparams,
    )
