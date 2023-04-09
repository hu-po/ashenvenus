import gc
import os
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def get_device(device: str = None):
    if device == None or device == "gpu":
        if torch.cuda.is_available():
            print("Using GPU")
            print('Clearing GPU memory')
            torch.cuda.empty_cache()
            gc.collect()
            return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


# Dataset Class
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
        image_mask_filename='mask.png',
        image_labels_filename='inklabels.png',
        slices_dir_filename='surface_volume',
        # Expected slices per fragment
        crop_size: Tuple[int] = (3, 68, 68),
        label_size: Tuple[int] = (256, 256),
        # Depth in scan is a Clipped Normal distribution
        min_depth: int = 0,
        max_depth: int = 65,
        avg_depth: float = 27.,
        std_depth: float = 10.,
        # Training vs Testing mode
        train: bool = True,
        # Device to use
        device: str = 'cuda',

    ):
        print(f'Making Dataset from {data_dir}')
        self.dataset_size = dataset_size
        self.points_per_crop = points_per_crop
        self.train = train
        self.device = device
        # Open Mask image
        _image_mask_filepath = os.path.join(data_dir, image_mask_filename)
        self.mask = np.array(cv2.imread(_image_mask_filepath, cv2.IMREAD_GRAYSCALE)).astype(np.bool)
        # Image dimmensions (depth, height, width)
        self.original_size = self.mask.shape
        self.crop_size = crop_size
        self.label_size = label_size
        # Open Label image
        if self.train:
            _image_labels_filepath = os.path.join(data_dir, image_labels_filename)
            self.labels = np.array(cv2.imread(_image_labels_filepath, cv2.IMREAD_GRAYSCALE)).astype(np.uint8)
        # Slices
        self.slice_dir = os.path.join(data_dir, slices_dir_filename)
        # Sample random crops within the image
        self.indices = np.zeros((dataset_size, 2, 3), dtype=np.int64)
        for i in range(dataset_size):
            # Select a random starting point for the subvolume
            start_depth = int(np.clip(np.random.normal(avg_depth, std_depth), min_depth, max_depth))
            start_height = np.random.randint(0, self.original_size[0] - self.crop_size[1])
            start_width = np.random.randint(0, self.original_size[1] - self.crop_size[2])
            self.indices[i, 0, :] = [start_depth, start_height, start_width]
            # End point is start point + crop size
            self.indices[i, 1, :] = [
                start_depth + self.crop_size[0],
                start_height + self.crop_size[1],
                start_width + self.crop_size[2],
            ]

    def __len__(self):
        return self.dataset_size
    
    def _make_pixel_stats(self):
        pass

    def __getitem__(self, idx):
        # Start and End points for the crop in pixel space
        start = self.indices[idx, 0, :]
        end = self.indices[idx, 1, :]
        # Load the relevant slices and pack into image tensor
        image = np.zeros(self.crop_size, dtype=np.float32)
        for i, _depth in enumerate(range(start[0], end[0])):
            _slice_filepath = os.path.join(self.slice_dir, f"{_depth:02d}.tif")
            _slice = np.array(cv2.imread(_slice_filepath, cv2.IMREAD_GRAYSCALE)).astype(np.float32)
            image[i, :, :] = _slice[start[1]: end[1], start[2]: end[2]]
        image = torch.from_numpy(image).to(device=self.device)

        # Choose Points within the crop for SAM to sample
        point_coords = np.zeros((self.points_per_crop, 2), dtype=np.int64)
        point_labels = np.zeros(self.points_per_crop, dtype=np.uint8)
        for i in range(self.points_per_crop):
            point_coords[i, 0] = np.random.randint(0, self.crop_size[1])
            point_coords[i, 1] = np.random.randint(0, self.crop_size[2])
            point_labels[i] = self.labels[
                start[1] + point_coords[i, 0],
                start[2] + point_coords[i, 1],
            ]
        point_coords = torch.from_numpy(point_coords).to(device=self.device)
        point_labels = torch.from_numpy(point_labels).to(device=self.device)

        if self.train:
            raw_label = self.labels[start[1]:end[1], start[2]:end[2]]
            label = cv2.resize(raw_label.astype(np.uint8), self.label_size, interpolation=cv2.INTER_NEAREST)
            label = torch.from_numpy(label).to(dtype=torch.float32)
            label = label.unsqueeze(0).clone().to(device=self.device)
            return image, point_coords, point_labels, label
        else:
            return image, point_coords, point_labels

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
    lr: float = 1e-4,
    wd: float = 1e-4,
    writer = None,
    # Dataset
    curriculum: str = "1",
    crop_size: Tuple[int] = (3, 68, 68),
    label_size: Tuple[int] = (1024, 1024),
    points_per_crop: int = 8,
    avg_depth: float = 27.,
    std_depth: float = 10.,
    **kwargs,
):
    device = get_device(device)  
    # TODO: Select only a subset of model parameters to train
    model = sam_model_registry[model](checkpoint=weights_filepath)
    # Turn off gradients for any of the image encoder
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    model.to(device=device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    step = 0
    best_score_dict: Dict[str, float] = {}
    for epoch in range(num_epochs):
        print(f"\n\n --- Epoch {epoch+1} of {num_epochs} --- \n\n")

        # Training
        for _dataset_id in curriculum:
            _dataset_filepath = os.path.join(train_dir, _dataset_id)
            print(f"Training on {_dataset_filepath} ...")
            _dataset = FragmentDataset(
                data_dir=_dataset_filepath,
                dataset_size=num_samples_train,
                points_per_crop = points_per_crop,
                crop_size = crop_size,
                label_size = label_size,
                avg_depth = avg_depth,
                std_depth = std_depth,
                train = True,
                device = device,
            )
            _dataloader = DataLoader(
                dataset=_dataset,
                batch_size=batch_size,
                shuffle=True,
                # pin_memory=True,
            )
            _loader = tqdm(_dataloader)
            for images, point_coords, point_labels, labels in _loader:
                if writer:
                    writer.add_images("input.image/train", images, step)
                    writer.add_images("input.label/train", labels*255, step)

                    # Plot point coordinates into a blank image of size images
                    _point_coords = point_coords.cpu().numpy()
                    _point_labels = point_labels.cpu().numpy()
                    _point_image = np.zeros((1, 3, images.shape[2], images.shape[3]), dtype=np.uint8)
                    point_width = 4
                    for i in range(_point_coords.shape[1]):
                        _height = _point_coords[0, i, 0]
                        _width = _point_coords[0, i, 1]
                        if _point_labels[0, i] == 0:
                            _point_image[0, 0, _height-point_width:_height+point_width, _width-point_width:_width+point_width] = 255
                        else:
                            _point_image[0, 1, _height-point_width:_height+point_width, _width-point_width:_width+point_width] = 255
                    writer.add_images(f"input.points/train/{_dataset_id}", _point_image, step)

                image_embeddings = model.image_encoder(images)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    # points=(point_coords, point_labels),
                    points=None,
                    boxes=None,
                    masks=labels,
                )
                # HACK: Something goes on here for batch sizes greater than 1
                # TODO: iou predictions could be used for additional loss?
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                if writer:
                    writer.add_images("output.masks/train", low_res_masks, step)
                loss = loss_fn(low_res_masks, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                _loss_name = f"{loss_fn.__class__.__name__}/train"
                _loader.set_postfix_str(f"{_loss_name}: {loss.item():.4f}")
                if writer:
                    writer.add_scalar(f"{_loss_name}", loss.item(), step)
            
        # Validation
        #   - Will overwrite the dataset and dataloader objects
        for _dataset_id in curriculum:
            _dataset_filepath = os.path.join(valid_dir, _dataset_id)
            print(f"Validating on dataset: {_dataset_filepath}")
            _dataset = FragmentDataset(
                data_dir=_dataset_filepath,
                dataset_size=num_samples_valid,
                points_per_crop = points_per_crop,
                crop_size = crop_size,
                label_size = label_size,
                avg_depth = avg_depth,
                std_depth = std_depth,
                train = True,
                device = device,
            )
            _dataloader = DataLoader(
                dataset=_dataset,
                batch_size=batch_size,
                shuffle=False,
                # pin_memory=True,
            )
            _score_name = f'score/valid/{_dataset_id}'
            if _score_name not in best_score_dict:
                best_score_dict[_score_name] = 0
            score = 0
            valid_step = 0
            _loader = tqdm(_dataloader)
            for images, point_coords, point_labels, labels in _loader:
                if writer:
                    valid_step += 1
                    writer.add_images(f"input.image/valid/{_dataset_id}", images, valid_step)
                    writer.add_images(f"input.label/valid/{_dataset_id}", labels*255, valid_step)

                    # Plot point coordinates into a blank image of size images
                    _point_coords = point_coords.cpu().numpy()
                    _point_labels = point_labels.cpu().numpy()
                    _point_image = np.zeros((1, 3, images.shape[2], images.shape[3]), dtype=np.uint8)
                    point_width = 4
                    for i in range(_point_coords.shape[1]):
                        _height = _point_coords[0, i, 0]
                        _width = _point_coords[0, i, 1]
                        if _point_labels[0, i] == 0:
                            _point_image[0, 0, _height-point_width:_height+point_width, _width-point_width:_width+point_width] = 255
                        else:
                            _point_image[0, 1, _height-point_width:_height+point_width, _width-point_width:_width+point_width] = 255
                    writer.add_images(f"input.points/valid/{_dataset_id}", _point_image, valid_step)

                image_embeddings = model.image_encoder(images)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    # points=(point_coords, point_labels),
                    points=None,
                    boxes=None,
                    masks=labels,
                )
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                if writer:
                    writer.add_images(f"output.masks/valid/{_dataset_id}", low_res_masks, valid_step)
                loss = loss_fn(low_res_masks, labels)
                score -= loss.item()

                _loss_name = f"{loss_fn.__class__.__name__}/valid/{_dataset_id}"
                _loader.set_postfix_str(f"{_loss_name}: {loss.item():.4f}")
                if writer:
                    writer.add_scalar(_loss_name, loss.item(), valid_step)
            
            # Overwrite best score if it is better
            score /= len(_dataloader)
            if score > best_score_dict[_score_name]:
                print(f"New best score! >> {score:.4f} (was {best_score_dict[_score_name]:.4f})")        
                best_score_dict[_score_name] = score
                if save_model:
                    _model_filepath = os.path.join(output_dir, f"model_{run_name}best_{_dataset_id}.pth")
                    print(f"Saving model to {_model_filepath}")
                    torch.save(model.state_dict(), _model_filepath)

        # Flush writer every epoch
        writer.flush()
    writer.close()
    return best_score_dict

def eval():
    pass

def eval_from_episode_dir():
    pass