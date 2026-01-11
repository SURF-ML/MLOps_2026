# from pathlib import Path
# from typing import Dict, Tuple

# from torch.utils.data import DataLoader
# from torchvision import transforms

# from .pcam import PCAMDataset

from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    # TODO: Define Transforms
    train_transform = None
    val_transform = None

    # TODO: Define Paths for X and Y (train and val)
    x_train = base_path / "camelyonpatch_level_2_split_train_x.h5"
    y_train = base_path / "camelyonpatch_level_2_split_train_y.h5"
    x_val = base_path / "camelyonpatch_level_2_split_valid_x.h5"
    y_val = base_path / "camelyonpatch_level_2_split_valid_y.h5"
    
    # TODO: Instantiate PCAMDataset for train and val
    train_dataset = PCAMDataset(str(x_train), str(y_train), transform=train_transform, filter_data=True)
    val_dataset = PCAMDataset(str(x_val), str(y_val), transform=val_transform, filter_data=False)

    with h5py.File(str(y_train), "r") as f:
        y_ds = f["y"] if "y" in f else list(f.values())[0]
        labels_all = np.array(y_ds[:]).squeeze().astype(int)

    labels = labels_all[train_dataset.indices]

    counts = np.bincount(labels, minlength=2)
    weights_per_class = 1.0 / counts
    sample_weights = weights_per_class[labels]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
    )


    # TODO: Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )
    
    return train_loader, val_loader
