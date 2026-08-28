"""PyTorch data pipeline for the Landslide4Sense dataset.

Serves (image, mask) pairs from the labelled TrainData, using the splits and
normalisation statistics produced by notebook 02_data_preprocessing.
"""

import json
from pathlib import Path

import h5py
import kagglehub
import numpy as np
import torch
from torch.utils.data import Dataset


def get_data_dir() -> Path:
    """Return the local path of the Landslide4Sense dataset (downloads if needed)."""
    return Path(kagglehub.dataset_download("tekbahadurkshetri/landslide4sense"))


class Landslide4SenseDataset(Dataset):
    """One split (train/val/test) of the labelled Landslide4Sense training data.

    Returns per item:
        image: float32 tensor of shape (14, 128, 128), z-scored per band
               with the training-split statistics
        mask:  float32 tensor of shape (128, 128), values 0/1
    """

    def __init__(self, data_dir: Path, split: str, splits_file: Path, stats_file: Path):
        splits = json.loads(Path(splits_file).read_text())
        if split not in splits:
            raise ValueError(f"split must be one of {sorted(splits)}, got {split!r}")
        self.ids = splits[split]

        stats = json.loads(Path(stats_file).read_text())
        self.mean = np.array(stats["mean"], dtype=np.float32)
        self.std = np.array(stats["std"], dtype=np.float32)

        self.img_dir = Path(data_dir) / "TrainData" / "img"
        self.mask_dir = Path(data_dir) / "TrainData" / "mask"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.ids[index]
        with h5py.File(self.img_dir / f"image_{image_id}.h5", "r") as f:
            img = f["img"][:].astype(np.float32)  # (128, 128, 14)
        with h5py.File(self.mask_dir / f"mask_{image_id}.h5", "r") as f:
            mask = f["mask"][:].astype(np.float32)  # (128, 128)

        img = (img - self.mean) / self.std
        image = torch.from_numpy(img).permute(2, 0, 1)  # (14, 128, 128)
        return image, torch.from_numpy(mask)
