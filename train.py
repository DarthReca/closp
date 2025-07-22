import datetime
import json
import os

import comet_ml
import comet_ml.integration
import comet_ml.integration.pytorch
import h5py
import hydra
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.utils.data as data
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageFile
from timm.utils import freeze, unfreeze
from torch import optim
from torchgeo.models import ResNet50_Weights, resnet50
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset import GeoCrisisDataModule
from models import GeoCLIP, LocGeoCLIP, RemoteCLIP, RGBGeoCLIP, SenCLIP, SkyScript

EPOCHS = 30
ImageFile.LOAD_TRUNCATED_IMAGES = True


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(args: DictConfig):
    seed_everything(42)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"checkpoints/{uuid}", exist_ok=True)

    comet_ml.login(offline_directory="comet-logs")
    experiment: comet_ml.Experiment = comet_ml.start(
        project_name="",
        workspace="",
        online=True,
        experiment_config=comet_ml.ExperimentConfig(name=uuid),
    )

    experiment.log_parameters(OmegaConf.to_container(args))

    dm = GeoCrisisDataModule(**args.dataset)
    dm.setup("fit")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_loc = nn.CrossEntropyLoss()

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()
    if args.get("skyclip", False):
        model = SkyScript(device).to(device)
    elif args.get("senclip", False):
        model = SenCLIP(device).to(device)
    elif args.get("remoteclip", False):
        model = RemoteCLIP(device=device).to(device)
    elif args.dataset.get("rgb_only", False):
        model = RGBGeoCLIP(mode="train").to(device)
    elif args.dataset.get("return_coords", False):
        model = LocGeoCLIP(**args.model).to(device)
    else:
        model = GeoCLIP(**args.model).to(device)
    experiment.set_model_graph(model)

    params = model.parameters()
    max_lrs = args.lr
    optimizer = optim.AdamW(params, lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        steps_per_epoch=len(iter(train_dataloader)),
        epochs=EPOCHS,
        div_factor=1e6,
        final_div_factor=1e6,
        pct_start=0.1,
    )

    best_val_loss = None
    step = 100
    satellite_loss = {"s1": 0, "s2": 0}
    satellite_batches = {"s1": 0, "s2": 0}
    previous_loss = float("inf")
    # finetuning contrastive like CLIP
    for epoch in range(EPOCHS):
        print("EPOCH", epoch + 1)
        # TRAIN
        with experiment.train():
            if hasattr(model, "mode"):
                model.mode = "train"
            model.train()
            progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            satellite_loss = {"s1": 0, "s2": 0}
            satellite_batches = {"s1": 0, "s2": 0}
            for i, (batch, batch_idx, dl_idx) in progress:
                optimizer.zero_grad()
                total_loss = 0
                for mini_batch in batch:
                    satellite = "s1" if mini_batch["image"].shape[1] == 2 else "s2"
                    inputs = {
                        "image": mini_batch["image"].to(device),
                        "input_ids": mini_batch["input_ids"].to(device),
                        "attention_mask": mini_batch["attention_mask"].to(device),
                    }
                    if "coords" in mini_batch:
                        inputs["coords"] = mini_batch["coords"].to(device)
                    outputs = model(**inputs, training=True)
                    # We extract only the last two or four outputs
                    outputs = outputs[len(outputs) // 2 :]
                    ground_truth = torch.arange(len(mini_batch["input_ids"])).to(device)
                    loss: list[torch.Tensor] = (
                        ls(o, ground_truth)
                        for o, ls in zip(outputs, [loss_img, loss_txt, loss_loc, loss_loc])
                    )
                    loss = sum(loss) / len(outputs)
                    satellite_loss[satellite] += loss.item()
                    satellite_batches[satellite] += 1
                    total_loss += loss

                total_loss /= len(batch)
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                if i % step == 0 and i > 0:
                    current_loss = sum(satellite_loss.values()) / sum(
                        satellite_batches.values()
                    )
                    progress.set_postfix(
                        {
                            "tl": current_loss,
                            "step": i,
                            "dec": current_loss < previous_loss,
                        }
                    )
                    for k, v in satellite_loss.items():
                        experiment.log_metric(
                            f"{k}_loss",
                            v / (satellite_batches[k] + 1e-8),
                            step=epoch * len(train_dataloader) + i,
                        )
                    for g, lr in enumerate(scheduler.get_last_lr()):
                        experiment.log_metric(
                            f"learning_rate_{g}",
                            lr,
                            step=epoch * len(train_dataloader) + i,
                        )
                    satellite_loss = {"s1": 0, "s2": 0}
                    satellite_batches = {"s1": 0, "s2": 0}
                    previous_loss = current_loss

        # VALIDATION
        with experiment.validate():
            if hasattr(model, "mode"):
                model.mode = "test"
            model.eval()
            satellite_loss = {"s1": 0, "s2": 0}
            satellite_batches = {"s1": 0, "s2": 0}
            with torch.no_grad():
                for i, (batch, batch_idx, dl_idx) in tqdm(enumerate(val_dataloader)):
                    satellite = "s1" if batch["image"].shape[1] == 2 else "s2"
                    inputs = {
                        "image": batch["image"].to(device),
                        "input_ids": batch["input_ids"].to(device),
                        "attention_mask": batch["attention_mask"].to(device),
                    }
                    if "coords" in batch:
                        inputs["coords"] = batch["coords"].to(device)
                    outputs = model(**inputs, training=False)
                    outputs = outputs[len(outputs) // 2 :]
                    ground_truth = torch.arange(len(batch["input_ids"])).to(device)
                    loss = (
                        ls(o, ground_truth)
                        for o, ls in zip(outputs, [loss_img, loss_txt, loss_loc])
                    )
                    loss = sum(loss) / len(outputs)
                    satellite_loss[satellite] += loss.item()
                    satellite_batches[satellite] += 1

            for k, v in satellite_loss.items():
                experiment.log_metric(
                    f"{k}_loss",
                    v / (satellite_batches[k] + 1e-8),
                    step=epoch * len(train_dataloader) + epoch * len(val_dataloader),
                )
            current_loss = sum(satellite_loss.values()) / sum(
                satellite_batches.values()
            )
            print("Validation Loss at epoch", epoch + 1, "->", current_loss)
            if best_val_loss is None or current_loss < best_val_loss:
                best_val_loss = current_loss
                torch.save(model.state_dict(), f"checkpoints/{uuid}/best_model.pth")
        experiment.log_epoch_end(epoch + 1)
    comet_ml.end()


if __name__ == "__main__":
    main()
