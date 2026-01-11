from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.
    """

    def __init__(self, x_path: str, y_path: str, transform: Optional[Callable] = None, filter_data: bool = False):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform

        # TODO: Initialize dataset
        # 1. Check if files exist
        if not self.x_path.exists():
            raise FileNotFoundError(f"X file not found: {self.x_path}")
        if not self.y_path.exists():
            raise FileNotFoundError(f"Y file not found: {self.y_path}")

        # 2. Open h5 files in read mode
        self.x_file = h5py.File(self.x_path, "r")
        self.y_file = h5py.File(self.y_path, "r")

        self.x_ds = self.x_file["x"] if "x" in self.x_file else list(self.x_file.values())[0]
        self.y_ds = self.y_file["y"] if "y" in self.y_file else list(self.y_file.values())[0]

        self.indices = np.arange(len(self.y_ds))

        if filter_data:
            x_all = np.clip(self.x_ds[:], 0, 255).astype(np.uint8)
            means = x_all.mean(axis=(1, 2, 3))
            self.indices = np.where((means > 5) & (means < 250))[0]

    def __len__(self) -> int:
        # TODO: Return length of dataset
        # The dataloader will know hence how many batches to create
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement data retrieval
        # 1. Read data at idx
        real_idx = int(self.indices[idx])
        x = self.x_ds[real_idx]
        y = self.y_ds[real_idx]

        # 2. Convert to uint8 (for PIL compatibility if using transforms)
        x = np.clip(x, 0, 255).astype(np.uint8)

        # 3. Apply transforms if they exist
        x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0

        if self.transform is not None:
            x = self.transform(x)

        # 4. Return tensor image and label (as long)
        y = int(np.array(y).squeeze())
        y = torch.tensor(y, dtype=torch.long)

        return x, y