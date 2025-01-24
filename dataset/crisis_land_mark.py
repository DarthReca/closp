import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import polars as pl
import xarray as xr
from lightning.pytorch.utilities import CombinedLoader
from pyproj import Proj
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import NonGeoDataset
from transformers import AutoTokenizer


class CrisisLandMarkDataModule(NonGeoDataModule):
    def __init__(self, root: str, batch_size: int = 32, num_workers: int = 0, **kwargs):
        super().__init__(
            CrisisLandMark,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.aug = nn.Identity()

    def setup(self, stage):
        if self.kwargs["satellite_type"] == "all":
            kwargs = {k: v for k, v in self.kwargs.items() if k != "satellite_type"}
            if stage in ["fit"]:
                self.train_dataset = [
                    self.dataset_class(  # type: ignore[call-arg]
                        split="train", satellite_type=s, **kwargs
                    )
                    for s in ["s2", "s1"]
                ]
            if stage in ["fit", "validate"]:
                self.val_dataset = [
                    self.dataset_class(  # type: ignore[call-arg]
                        split="val", satellite_type=s, **kwargs
                    )
                    for s in ["s2", "s1"]
                ]
            if stage in ["test"]:
                self.test_dataset = [
                    self.dataset_class(  # type: ignore[call-arg]
                        split="test", satellite_type=s, **kwargs
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


class CrisisLandMark(NonGeoDataset):
    satellite_datasets = {
        "s2": ["benv2s2", "cabuar", "sen2flood"],
        "s1": ["benv2s1", "mmflood", "sen1flood", "quakeset"],
        "all": ["benv2s1", "benv2s2", "cabuar", "mmflood", "sen1flood", "sen2flood"],
    }
    text_encoders = {
        "MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "openclip": "RN50",
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
        rgb_only=False,
        tokenizer: str | None = None,
        context_length: int = 77,
        satellite_type: str = "all",
        return_key: bool = False,
        return_coords: bool = False,
        augment_labels: bool = False,
    ):
        """
        Args:
            root (str): Root directory where the dataset is stored.
            split (str, optional): The dataset split, supports ``train``, and ``test``.
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        """
        assert split in ["train", "val", "test"]
        self.rgb_only = rgb_only
        self.return_key = return_key
        self.return_coords = return_coords
        self.tokenizer = None
        if tokenizer and tokenizer != "openclip":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.text_encoders.get(tokenizer, tokenizer),
                use_fast=True,
                cache_dir="cache",
            )
        elif tokenizer == "openclip":
            self.tokenizer = open_clip.get_tokenizer(self.text_encoders["openclip"])

        self.samples = pl.scan_parquet(f"{root}/geocrisis.parquet")
        fraction = 0.002 if split == "val" else 1.0
        split = "training" if split == "train" else "corpus"
        keys = pl.read_parquet(f"{root}/splits.parquet").filter(split=split)["key"]

        self.samples = (
            self.samples.filter(pl.col("key").is_in(keys))
            .explode("labels")
            .with_columns(
                pl.col("labels").str.to_lowercase(),
                original_labels=pl.col("labels"),
            )
        )
        self.samples = (
            self.samples.group_by("file", "key")
            .agg("labels", "original_labels")
            .with_columns(
                pl.col("labels").list.join(". ").str.to_lowercase(),
                pl.col("original_labels").list.join(". ").str.to_lowercase(),
            )
        ).filter(
            pl.col("key").str.contains_any(self.satellite_datasets[satellite_type])
        )

        self.samples = (
            self.samples.sort("key").collect().sample(fraction=fraction, seed=42)
        )
        if self.tokenizer is not None:
            if isinstance(self.tokenizer, open_clip.SimpleTokenizer):
                batch_encoding = {
                    "input_ids": self.tokenizer(
                        self.samples["labels"].to_list()
                    ).numpy(),
                    "attention_mask": np.ones((len(self.samples), 77)),
                }
            else:
                batch_encoding = self.tokenizer(
                    self.samples["labels"].to_list(),
                    padding="max_length",
                    truncation=True,
                    max_length=context_length,
                    return_tensors="np",
                )
            self.samples = self.samples.with_columns(
                input_ids=batch_encoding["input_ids"],
                attention_mask=batch_encoding["attention_mask"],
            )

    def __len__(self):
        return self.samples.height

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        sample = self.samples.row(idx, named=True)
        path = sample["file"]
        key = sample["key"]
        with h5py.File(path, "r") as f:
            data = f[key]["image"][:]
            center = f[key]["coords"][:, 60, 60]
            crs = f[key].attrs["crs"]
        if any([x in path for x in ["benv2s1", "sen1flood"]]):
            data = np.flip(data, axis=0)
        if "quakeset" in path:
            data = data[2:]
        # Add a zero-ed channel for S2 band 10
        if data.shape[0] == 12:
            data = np.insert(data, 10, 0, axis=0)
            data = np.nan_to_num(data, nan=0)
        else:
            data = np.nan_to_num(data, nan=1e-6)
        # Extract RGB bands if needed
        if self.rgb_only:
            if data.shape[0] == 12:
                data = data[(3, 2, 1), :]
            else:
                data = np.stack([data[0], data[1], data[0] - data[1]])
        tokenization_results = {}
        if self.tokenizer is not None:
            tokenization_results["attention_mask"] = np.array(sample["attention_mask"])
            tokenization_results["input_ids"] = np.array(sample["input_ids"])
        if self.return_key:
            tokenization_results |= {"key": key}
        if self.return_coords:
            tokenization_results |= {
                "coords": np.array(
                    Proj(crs)(center[0], center[1], inverse=True), dtype=np.float32
                )
            }
        return {
            "image": np.ascontiguousarray(data),
            "labels": sample.get("labels"),
        } | tokenization_results

    def plot(self, image, ax=None):
        rgb = image
        if image.shape[0] in (12, 13):
            rgb = image[(3, 2, 1), :] / 10000 * 5
        elif image.shape[0] == 2:
            rgb = np.stack([image[0], image[1], image[0] - image[1]])
            rgb = rgb[0]
        return xr.DataArray(rgb).plot.imshow(
            figsize=(10, 10),
            robust=True,
            ax=ax,
            cmap=None if rgb.shape[0] == 3 else "gray",
            add_colorbar=False,
        )
