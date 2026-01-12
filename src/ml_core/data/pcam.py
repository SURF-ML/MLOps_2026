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
        # 2. Open h5 files in read mode

        # 1. Check if files exist
        if not self.x_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(
                f"PCAM files not found at {self.x_path} or {self.y_path}"
            )

        # 2. Open h5 files in read mode (lazy loading)
        self._x_h5 = h5py.File(self.x_path, "r")
        self._y_h5 = h5py.File(self.y_path, "r")

        # Prefer "x"/"y" keys, else fall back to first key
        x_key = "x" if "x" in self._x_h5 else list(self._x_h5.keys())[0]
        y_key = "y" if "y" in self._y_h5 else list(self._y_h5.keys())[0]

        self.x_data = self._x_h5[x_key]
        self.y_data = self._y_h5[y_key]

        # Optional heuristic filtering (black/white artifacts)
        n = len(self.x_data)
        self.indices = np.arange(n, dtype=np.int64)
        if filter_data:
            keep = []
            for i in range(n):
                m = float(np.mean(self.x_data[i]))
                if 10.0 < m < 245.0:
                    keep.append(i)
            self.indices = np.asarray(keep, dtype=np.int64)


    def __len__(self) -> int:
        # TODO: Return length of dataset
        # The dataloader will know hence how many batches to create
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement data retrieval
        # 1. Read data at idx
        # 2. Convert to uint8 (for PIL compatibility if using transforms)
        # 3. Apply transforms if they exist
        # 4. Return tensor image and label (as long)
        real_idx = int(self.indices[idx])

        image = self.x_data[real_idx]
        label = self.y_data[real_idx]

        # Numerical stability: clip before casting
        image = np.clip(image, 0, 255).astype(np.uint8)

        if self.transform is not None:
            image_t = self.transform(image)
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0

        label_t = torch.as_tensor(label).squeeze().to(torch.long)
        return image_t, label_t

