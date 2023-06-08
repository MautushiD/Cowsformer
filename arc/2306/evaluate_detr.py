# native imports
import os
import numpy as np

# local imports
from data.datasets import CocoDetection, CocoPanoptic
from detr.datasets.coco_eval import CocoEvaluator
from detr.datasets.panoptic_eval import PanopticEvaluator

# torch
from transformers import DetrFeatureExtractor, DetrConfig, DetrForSegmentation
import torch
from torch.utils.data import DataLoader


DIR_DATA = os.path.join("data", "panoptic_val2017")
dir_img = os.path.join(DIR_DATA, "images")
dir_ann = os.path.join(DIR_DATA, "annotation_mask")
jsn_ann = os.path.join(DIR_DATA, "annotation.json")
ins_ann = os.path.join(DIR_DATA, "instances_val2017.json")

# feature extractor
feature_extractor = DetrFeatureExtractor.from_pretrained(
    "facebook/detr-resnet-50-panoptic", size=500, max_size=600
)

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


dataset = CocoPanoptic(
    img_folder=dir_img,
    ann_folder=dir_ann,
    ann_file=jsn_ann,
    feature_extractor=feature_extractor,
)
indices = np.random.randint(low=0, high=len(dataset), size=50)
train_dataset = torch.utils.data.Subset(dataset, indices[:40])
val_dataset = torch.utils.data.Subset(dataset, indices[40:])
val_dataloader = DataLoader(
    val_dataset,
    collate_fn=collate_fn,
    batch_size=1,
    num_workers=4,
)

# load ground truths
base_ds = CocoDetection(img_folder=dir_img,
                        ann_file=ins_ann,
                        feature_extractor=feature_extractor, train=False).coco

iou_types = ['bbox', 'segm']
coco_evaluator = CocoEvaluator(base_ds, iou_types) # initialize evaluator with ground truths



# this section is only needed if you want to create a new annotation file
import json

# read in all annotations (5000)
with open(jsn_ann) as f:
  data_ann = json.load(f)

# get image ids of images in validation set
image_ids = []
for batch in val_dataloader:
  labels = batch['labels']
  for label in labels:
    image_ids.append(label['image_id'].item())
print(image_ids)

# only keep those annotations
relevant_annotations = []
for ann in data_ann['annotations']:
  if ann['image_id'] in image_ids:
    relevant_annotations.append(ann)

# replace 5000 annotation with only 50
data_ann['annotations'] = relevant_annotations

# save this as a new json annotation file
path_val = os.path.join("data", "panoptic_val2017", 'annotation_val.json')
with open(path_val, 'w') as f:
    json.dump(data_ann, f)

# section end------------------------------------------------------------
# inititialiaze panoptic evaluator with the ground truth annotations
panoptic_evaluator = PanopticEvaluator(
            path_val,
            val_dataloader.dataset.dataset.ann_folder,
            output_dir=".",
        )


from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load pytorch_model.bin
model = DetrForSegmentation.from_pretrained("detr_panoptic")


model.to(device)
model.eval()

print("Running evaluation...")

for idx, batch in enumerate(tqdm(val_dataloader)):
    # get the inputs
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized

    # forward pass
    with torch.no_grad():
      outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    # # object detection evaluation
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    # results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
    # target_sizes = torch.stack([t["size"] for t in labels], dim=0)
    # results = feature_extractor.post_process_segmentation(outputs, target_sizes)
    # res = {target['image_id'].item(): output for target, output in zip(labels, results)}
    # coco_evaluator.update(res)

    # panoptic segmentation evaluation
    target_sizes = torch.stack([t["size"] for t in labels], dim=0)
    results_panoptic = feature_extractor.post_process_panoptic(outputs, target_sizes, orig_target_sizes) # convert outputs of model to COCO api
    for i, target in enumerate(labels):
        image_id = target["image_id"].item()
        file_name = f"{image_id:012d}.png"
        results_panoptic[i]["image_id"] = image_id
        results_panoptic[i]["file_name"] = file_name
    panoptic_evaluator.update(results_panoptic)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()