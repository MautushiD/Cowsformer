# native imports
from importlib.resources import path
from pathlib import Path
import json
import os
import numpy as np
from PIL import Image

# torch
import torch
from torch.utils.data import DataLoader
import torchvision
from transformers import DetrFeatureExtractor


class CocoPanoptic(torch.utils.data.Dataset):
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
    def __init__(self, img_folder, ann_folder, ann_file, feature_extractor):
        with open(ann_file, "r") as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco["images"] = sorted(self.coco["images"], key=lambda x: x["id"])
        # sanity check
        if "annotations" in self.coco:
            for img, ann in zip(self.coco["images"], self.coco["annotations"]):
                assert img["file_name"][:-4] == ann["file_name"][:-4]

        self.img_folder = img_folder
        self.ann_folder = Path(ann_folder)
        self.ann_file = ann_file
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        ann_info = (
            self.coco["annotations"][idx]
            if "annotations" in self.coco
            else self.coco["images"][idx]
        )
        img_path = Path(self.img_folder) / ann_info["file_name"].replace(".png", ".jpg")

        img = Image.open(img_path).convert("RGB")

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        encoding = self.feature_extractor(
            images=img,
            annotations=ann_info,
            masks_path=self.ann_folder,
            return_tensors="pt",
        )
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target

    def __len__(self):
        return len(self.coco["images"])

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, feature_extractor, train=True):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target
    


# # pick the 3rd image
# pixel_values, target = train_dataset[2]
# print(pixel_values.shape)
# print(target.keys())
# print("Number of training examples:", len(train_dataset))
# print("Number of validation examples:", len(val_dataset))
