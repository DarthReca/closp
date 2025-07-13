from typing import Literal

import kornia.augmentation as K
import open_clip
import torch
import torch.nn as nn
from timm import create_model, models
from torchgeo.models import (
    ResNet50_Weights,
    Swin_V2_B_Weights,
    ViTSmall16_Weights,
    resnet50,
    swin_v2_b,
    vit_small_patch16_224,
)
from torchvision.transforms.functional import to_tensor
from transformers import AutoModel, AutoTokenizer

from .location_encoder import satclip_location_encoder


def vit_large_patch16_224(weights_dict: dict):
    weights = torch.load(weights_dict["path"], map_location="cpu")
    model: torch.nn.Module = create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,
        in_chans=weights_dict["in_chans"],
    )
    assert len(model.load_state_dict(weights["model"], strict=False).missing_keys) == 0
    return model


def vit_small_patch16_224_v2(weights_dict: dict):
    weights = torch.load(weights_dict["path"], map_location="cpu")
    model = create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=0,
        in_chans=weights_dict["in_chans"],
    )
    assert len(model.load_state_dict(weights["model"], strict=False).missing_keys) == 0
    return model


class CLOSP(torch.nn.Module):
    vision_models = {
        "resnet50": (
            resnet50,
            resnet50,
            ResNet50_Weights.SENTINEL1_ALL_MOCO,
            ResNet50_Weights.SENTINEL2_ALL_MOCO,
        ),
        "vit-s": (
            vit_small_patch16_224_v2,
            vit_small_patch16_224,
            {"in_chans": 2, "path": "weights/ssl4eo/B2_vits16_mae_ep99.pth"},
            ViTSmall16_Weights.SENTINEL2_ALL_MOCO,
        ),
        "vit-l": (
            vit_large_patch16_224,
            vit_large_patch16_224,
            {"in_chans": 2, "path": "weights/ssl4eo/B2_vitl16_mae_ep99.pth"},
            {"in_chans": 13, "path": "weights/ssl4eo/B13_vitl16_mae_ep99.pth"},
        ),
        "mix": (
            resnet50,
            vit_small_patch16_224,
            ResNet50_Weights.SENTINEL1_ALL_MOCO,
            ViTSmall16_Weights.SENTINEL2_ALL_MOCO,
        ),
    }

    def __init__(
        self,
        text_model: str,
        vision_model: str,
        scale: float = 1.0,
        freeze_pretrained: bool = False,
    ):
        assert vision_model in self.vision_models
        super(CLOSP, self).__init__()
        s1_vision, s2_vision, s1_weights, s2_weights = self.vision_models[vision_model]
        self.s1_encoder = s1_vision(s1_weights)
        self.s2_encoder = s2_vision(s2_weights)

        self.text_model = AutoModel.from_pretrained(text_model).to("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)

        with torch.no_grad():
            out_shape = self.text_model(
                torch.ones(1, 64, dtype=torch.long).long()
            ).last_hidden_state.shape[-1]
            in_s2 = self.s2_encoder(torch.randn(1, 13, 224, 224)).shape[-1]
            in_s1 = self.s1_encoder(torch.randn(1, 2, 224, 224)).shape[-1]

        self.s2_projection = nn.Linear(in_s2, out_shape)
        self.s1_projection = nn.Linear(in_s1, out_shape)
        self.scale = scale

        s2_transform = [K.Normalize(mean=0, std=10000)]
        s1_transform = []
        if isinstance(self.s2_encoder, models.VisionTransformer):
            s2_transform += [K.Resize(224)]
        if isinstance(self.s1_encoder, models.VisionTransformer):
            s1_transform += [K.Resize(224)]
        self.val_s2_transform = K.AugmentationSequential(
            *s2_transform,
            data_keys=["image"],
        )
        self.val_s1_transform = K.AugmentationSequential(
            *s1_transform,
            data_keys=["image"],
        )
        self.train_transform = K.AugmentationSequential(
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            # K.RandomSaltAndPepperNoise(),
            data_keys=["image"],
        )

        if freeze_pretrained:
            self.s1_encoder.requires_grad_(False)
            self.s2_encoder.requires_grad_(False)
            self.text_model.requires_grad_(False)

    @property
    def device(self):
        return self.s2_projection.weight.device

    def freeze_pretrained(self, freeze: bool = True):
        self.s1_encoder.requires_grad_(not freeze)
        self.s2_encoder.requires_grad_(not freeze)
        self.text_model.requires_grad_(not freeze)

    def forward(
        self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        image = image.float()
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()

        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden_state = text_output.hidden_states[-1]
        text_embed = last_hidden_state[:, 0, :]

        if image.shape[1] == 2:
            image = self.val_s1_transform(image)
            image = self.train_transform(image)
            image_embed = self.s1_projection(self.s1_encoder(image))
        else:
            image = self.val_s2_transform(image)
            image = self.train_transform(image)
            image_embed = self.s2_projection(self.s2_encoder(image))

        image_embed = F.normalize(image_embed, p=2, dim=-1)
        text_embed = F.normalize(text_embed, p=2, dim=-1)

        logits_per_image = self.scale * image_embed @ text_embed.T
        logits_per_text = self.scale * text_embed @ image_embed.T

        return image_embed, text_embed, logits_per_image, logits_per_text

    def encode_image(self, image: torch.FloatTensor):
        satellite = "s1" if image.shape[1] == 2 else "s2"
        if satellite == "s1":
            image = self.val_s1_transform(image)
            embs = self.s1_projection(self.s1_encoder(image))
        else:
            image = self.val_s2_transform(image)
            embs = self.s2_projection(self.s2_encoder(image))
        return embs

    def encode_text(
        self,
        text: str = None,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.LongTensor = None,
    ):
        if text is not None:
            batch_encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            input_ids = batch_encoding["input_ids"].to(self.device)
            attention_mask = batch_encoding["attention_mask"].to(self.device)

        text_output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden_state = text_output.hidden_states[-1]
        text_embed = last_hidden_state[:, 0, :]
        return text_embed


class GeoCLOSP(CLOSP):
    def __init__(
        self,
        text_model: str,
        vision_model: str,
        location_encoder: str,
        scale=1,
        freeze_pretrained=False,
    ):
        super().__init__(text_model, vision_model, scale, freeze_pretrained)
        self.location_encoder = satclip_location_encoder(location_encoder)
        self.location_projection = nn.Linear(
            self.location_encoder.nnet.last_layer.dim_out,
            self.s1_projection.out_features,
        )
        if self.freeze_pretrained:
            self.location_encoder.requires_grad_(False)

    def freeze_pretrained(self, freeze=True):
        super().freeze_pretrained(freeze)
        self.location_encoder.requires_grad_(not freeze)

    def forward(self, image, input_ids, attention_mask, coords):
        image_embed, text_embed, logits_per_image, logits_per_text = super().forward(
            image, input_ids, attention_mask
        )

        location_embed = self.location_encoder(coords)
        location_embed = self.location_projection(location_embed)

        location_embed = F.normalize(location_embed, p=2, dim=-1)

        logits_per_loc_img = self.scale * location_embed @ image_embed.T
        logits_per_img_loc = self.scale * image_embed @ location_embed.T

        return (
            image_embed,
            text_embed,
            location_embed,
            logits_per_image,
            logits_per_text,
            logits_per_loc_img,
            logits_per_img_loc,
        )
