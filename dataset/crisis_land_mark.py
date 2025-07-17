import os
from typing import Callable, Literal

import h5py
import hdf5plugin
import pandas as pd
import torch
from lightning.pytorch.utilities import CombinedLoader
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import NonGeoDataset

from .mapping import CORINE_TO_DW


class CrisisLandMarkDataModule(NonGeoDataModule):
    def __init__(self, root: str, batch_size: int = 32, num_workers: int = 0, **kwargs):
        super().__init__(
            CrisisLandMarkDataset,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.aug = nn.Identity()

    def setup(self, stage: str):
        if self.kwargs["satellite"] == "all":
            kwargs = {k: v for k, v in self.kwargs.items() if k != "satellite"}
            if stage in ["fit"]:
                self.train_dataset = [
                    self.dataset_class(  # type: ignore[call-arg]
                        split="train", satellite=s, **kwargs
                    )
                    for s in ["s2", "s1"]
                ]
            if stage in ["test"]:
                self.test_dataset = [
                    self.dataset_class(  # type: ignore[call-arg]
                        split="test", satellite=s, **kwargs
                    )
                    for s in ["s2", "s1"]
                ]
        else:
            super().setup(stage)

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        datasets = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        if not isinstance(datasets, list):
            datasets = [datasets]
        loaders = [
            DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=split == "train",
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                persistent_workers=self.num_workers > 0,
            )
            for dataset in datasets
        ]
        return CombinedLoader(
            loaders, mode="sequential" if split != "train" else "max_size_cycle"
        )


class CrisisLandMarkDataset(NonGeoDataset):
    satellite_datasets = {
        "s2": ["benv2s2", "cabuar", "sen2flood"],
        "s1": ["benv2s1", "mmflood", "sen1flood", "quakeset"],
    }

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"],
        transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
        | None = None,
        download: bool = False,
        satellite: Literal["s2", "s1"] = "s2",
    ):
        super().__init__()

        # --- 1. Validate inputs ---
        if split not in ["train", "test"]:
            raise ValueError(f"Split must be 'train' or 'test', but got {split}")

        self.root_dir = root
        self.split = split
        self.transform = transform

        # --- 2. Define file paths ---
        self.h5_path = os.path.join(self.root_dir, "crisislandmark.h5")
        self.metadata_path = os.path.join(self.root_dir, "metadata.parquet")

        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"crisislandmark.h5 not found at {self.h5_path}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"metadata.parquet not found at {self.metadata_path}"
            )

        # --- 3. Load metadata and filter for the correct split ---
        metadata_df = pd.read_parquet(self.metadata_path)

        # Filter the sample keys based on the desired split
        metadata_df = metadata_df[metadata_df["split"] == self.split]
        metadata_df = metadata_df[
            metadata_df["key"].str.contains(
                "|".join(self.satellite_datasets[satellite]), regex=True
            )
        ]
        metadata_df = metadata_df[["key", "labels"]].explode("labels")

        lowered_map = {k.lower(): v.lower() for k, v in CORINE_TO_DW.items()}
        lowered_map |= {k: k.lower() for k in metadata_df["labels"].unique()}
        metadata_df["labels"] = metadata_df["labels"].str.lower().map(lowered_map)
        metadata_df = metadata_df.groupby("key").agg(set)
        self.sample_keys = metadata_df.to_records(index=True)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.sample_keys)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | list[str]]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image, coordinates, labels, and other metadata.
                  The image and coordinates are returned as PyTorch tensors.
        """
        # Get the unique key for the sample
        key, labels = self.sample_keys[idx]

        with h5py.File(self.h5_path, "r") as f:
            sample_group = f[key]

            # Read data arrays
            image_np = sample_group["image"][:]
            coords_np = sample_group["coords"][:]

            # Read metadata attributes
            try:
                crs = sample_group.attrs["crs"]
                timestamp = sample_group.attrs["timestamp"]
            except KeyError as e:
                print(f"Missing attribute in {key}: {e}")
                raise KeyError(f"Missing attribute in {key}: {e}") from e

        # --- Convert NumPy arrays to PyTorch tensors ---
        image_tensor = torch.from_numpy(image_np).float()
        coords_tensor = torch.from_numpy(coords_np).float()

        # --- Assemble the sample dictionary ---
        sample = {"image": image_tensor, "coords": coords_tensor}

        # --- Apply transforms if any ---
        if self.transform:
            sample = self.transform(sample)

        # --- Add metadata ---
        sample |= {"labels": ". ".join(labels), "crs": crs, "timestamp": timestamp}

        return sample
