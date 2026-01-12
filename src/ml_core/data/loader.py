from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

from .pcam import PCAMDataset


def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    # TODO: Define Transforms
    # train_transform = ...
    # val_transform = ...
    train_transform = transforms.Compose(
    	[
        	transforms.ToPILImage(),
        	transforms.ToTensor(),
        	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    	]
	)
    val_transform = transforms.Compose(
    	[
        	transforms.ToPILImage(),
        	transforms.ToTensor(),
        	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    	]
	)



    # TODO: Define Paths for X and Y (train and val)
    x_train = base_path / "camelyonpatch_level_2_split_train_x.h5"
    y_train = base_path / "camelyonpatch_level_2_split_train_y.h5"
    x_val = base_path / "camelyonpatch_level_2_split_valid_x.h5"
    y_val = base_path / "camelyonpatch_level_2_split_valid_y.h5"

    # TODO: Instantiate PCAMDataset for train and val
    train_dataset = PCAMDataset(str(x_train), str(y_train), transform=train_transform)
    val_dataset = PCAMDataset(str(x_val), str(y_val), transform=val_transform)

    # TODO: Create DataLoaders
    # train_loader = ...
    # val_loader = ...

    batch_size = int(data_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))

    # --- WeightedRandomSampler to handle class imbalance ---
    torch = __import__("torch")
    np = __import__("numpy")
    h5py = __import__("h5py")

    # Read labels from the train y.h5 file
    with h5py.File(str(y_train), "r") as f:
        y_key = "y" if "y" in f else list(f.keys())[0]
        y = np.asarray(f[y_key][:]).squeeze().astype(np.int64)

    class_counts = np.bincount(y)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,      # sampler replaces shuffle
        num_workers=num_workers,
    )


    val_loader = DataLoader(
    	val_dataset,
    	batch_size=batch_size,
    	shuffle=False,
    	num_workers=num_workers,
    )


    return train_loader, val_loader
