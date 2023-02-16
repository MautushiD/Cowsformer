#https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForSegmentation_on_custom_dataset_end_to_end_approach.ipynb#scrollTo=J-ma9DPmL9t_

# native
import os
import numpy as np

# transformers/torch
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DetrFeatureExtractor, DetrConfig, DetrForSegmentation
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# local
from data.datasets import CocoPanoptic


class DetrPanoptic(pl.LightningModule):
    def __init__(self, model, lr, lr_backbone, weight_decay, loader):
        super().__init__()

        self.model = model

        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.loader = loader

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
        )

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )

        return optimizer

    def train_dataloader(self):
        return self.loader["train"]

    def val_dataloader(self):
        return self.loader["val"]


# create dataloader
# define collate function
def collate_fn(batch):
    feature_extractor = DetrFeatureExtractor.from_pretrained(
        "facebook/detr-resnet-50-panoptic", size=500, max_size=600
    )
    pixel_values = [item[0] for item in batch]
    encoded_input = feature_extractor.pad_and_create_pixel_mask(
        pixel_values, return_tensors="pt"
    )
    labels = [item[1] for item in batch]
    batch = {}
    batch["pixel_values"] = encoded_input["pixel_values"]
    batch["pixel_mask"] = encoded_input["pixel_mask"]
    batch["labels"] = labels
    return batch


if __name__ == "__main__":
    # model definition ------------------------------------------------------------
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
    state_dict = model.state_dict()
    # Remove class weights
    del state_dict["detr.class_labels_classifier.weight"]
    del state_dict["detr.class_labels_classifier.bias"]
    # define new model with custom class classifier
    config = DetrConfig.from_pretrained(
        "facebook/detr-resnet-50-panoptic", num_labels=250
    )
    model = DetrForSegmentation(config)
    # load config and state dict
    model.load_state_dict(state_dict, strict=False)

    # data definition -------------------------------------------------------------
    ROOT = os.path.join("data", "panoptic_val2017")
    dir_img = os.path.join(ROOT, "images")
    dir_ann = os.path.join(ROOT, "annotation_mask")
    jsn_ann = os.path.join(ROOT, "annotation.json")

    # feature extractor
    feature_extractor = DetrFeatureExtractor.from_pretrained(
        "facebook/detr-resnet-50-panoptic", size=500, max_size=600
    )
    # ini coco dataset
    dataset = CocoPanoptic(
        img_folder=dir_img,
        ann_folder=dir_ann,
        ann_file=jsn_ann,
        feature_extractor=feature_extractor,
    )

    # split dataset
    indices = np.random.randint(low=0, high=len(dataset), size=50)
    train_dataset = torch.utils.data.Subset(dataset, indices[:40])
    val_dataset = torch.utils.data.Subset(dataset, indices[40:])

    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=3, num_workers=4, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=4,
    )
    loader = dict()
    loader["train"] = train_dataloader
    loader["val"] = val_dataloader

    # training definition ---------------------------------------------------------
    model = DetrPanoptic(
        model=model, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, loader=loader
    )

    trainer = pl.Trainer(max_epochs=2, gradient_clip_val=0.1, accelerator='gpu', devices=1)
    trainer.fit(model)

    # save model
    model.model.save_pretrained("detr_panoptic")
