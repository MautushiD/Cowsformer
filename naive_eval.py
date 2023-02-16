import os
import sys

# native imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

# local imports
from models.device import get_device
from models.models import init_model, get_labels, get_features_ext

# torch imports
from datasets import load_dataset


if __name__ == "__main__":
    # CONSTANTS
    MODEL_NAME = "facebook/detr-resnet-50-panoptic"
    PATH_DATA = os.path.join("data", "cow50", "cvat")
    PATH_OUT = os.path.join("out", "cow50.csv")
    COW_LB_ID = 21
    DEVICE = get_device()

    # model
    model = init_model(MODEL_NAME)
    model.to(DEVICE)
    feature_extractor = get_features_ext(MODEL_NAME)

    # dataset
    dataset = load_dataset(os.path.join(PATH_DATA, "data.py"), split="train")

    filenames = []
    obs = []
    pre = []
    for data in dataset:
        filename = data["filename"]
        image = np.array(data["image"])
        count = data["count"]
        # encoding contrains: pixel_values and pixel_mask (all ones)
        encoding = feature_extractor(image, return_tensors="pt")
        encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

        # forward pass
        outputs = model(**encoding)

        # post-processing
        result = feature_extractor.post_process_panoptic_segmentation(outputs, overlap_mask_area_threshold=.4)[0]
        seg_info = result["segments_info"]
        pred_count = np.sum([seg["label_id"] == COW_LB_ID for seg in seg_info])

        # write
        filenames.append(filename)
        obs.append(count)
        pre.append(pred_count)

    pd.DataFrame({"filename": filenames, "obs": obs, "pre": pre}).to_csv(PATH_OUT, index=False)

# processed_sizes = torch.as_tensor(image.shape[:2]).unsqueeze(0)
# result_old = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]
# result_old["segments_info"]



