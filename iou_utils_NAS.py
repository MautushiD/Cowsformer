import os
import numpy as np
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt
from super_gradients.training import models
from torchinfo import summary
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
import cv2


dataset_params = {
    "data_dir": "data_directory",
    "train_images_dir": "train/images",
    "train_labels_dir": "train/labels",
    "val_images_dir": "valid/images",
    "val_labels_dir": "valid/labels",
    "test_images_dir": "test/images",
    "test_labels_dir": "test/labels",
    "classes": ["cow"],
}


def save_predict_images_from_dir(model, image_dir, output_dir, conf=0.60, show=False):
    # Get a list of all image files in the directory
    image_files = list(Path(image_dir).rglob("*.jpg"))

    # Loop over all images and make predictions
    predictions = []
    for image_file in image_files:
        # Load image
        img = Image.open(image_file)

        # Make prediction

        if show == True:
            results = model.predict(img, conf).show()
        else:
            results = model.predict(img, conf).save(output_dir)
        # results.save()  # Save the image with bounding boxes
        predictions.append(results)

    return predictions


def show_predicted_images_from_dir(model, image_dir, conf=0.60):
    # Get a list of all image files in the directory
    image_files = list(Path(image_dir).rglob("*.jpg"))

    # Loop over all images and make predictions
    predictions = []
    for image_file in image_files:
        # Load image
        img = Image.open(image_file)

        # Make prediction

        # if show == True:
        # results = model.predict(img, conf).show()
        # else:
        # results = model.predict(img, conf)
        model.predict(img, conf=conf).show()
        # results.save()  # Save the image with bounding boxes
        # predictions.append(results)

    # return predictions


def cxcyxy_to_xyxy(image_label_txt, image_path):
    # Open image and get dimensions
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Read label file
    with open(image_label_txt, "r") as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(
            float, line.strip().split())

        # Convert to x_min, y_min, x_max, y_max format
        x_min = (x_center - width / 2) * image_width
        y_min = (y_center - height / 2) * image_height
        x_max = (x_center + width / 2) * image_width
        y_max = (y_center + height / 2) * image_height

        boxes.append([x_min, y_min, x_max, y_max])

    return boxes


def get_boxes_xyxy(model, image_path, conf=0.6):
    # Assuming model.predict returns a single ImageDetectionPrediction object
    # for the given image.
    result = model.predict(image_path, conf)

    # Assuming result contains an attribute that directly gives us the bounding boxes,
    # which might be named differently. Replace `bboxes_xyxy` with the correct attribute name.
    all_boxes = []
    if hasattr(result, 'prediction') and hasattr(result.prediction, 'bboxes_xyxy'):
        # Directly accessing bounding boxes from the result's prediction attribute.
        bboxes = result.prediction.bboxes_xyxy
        for bbox in bboxes:
            all_boxes.append(bbox.tolist())
    else:
        # Handle the case where the expected attributes are not present.
        print("The prediction object does not have the expected structure.")

    return all_boxes



def box_area(box_xyxy):
    x_min, y_min, x_max, y_max = box_xyxy
    return (x_max - x_min) * (y_max - y_min)


def bbox_iou(box1_xyxy, box2_xyxy):
    x1, y1, x1_max, y1_max = box1_xyxy
    x2, y2, x2_max, y2_max = box2_xyxy

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = box_area(box1_xyxy)
    box2_area = box_area(box2_xyxy)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def draw_boxes_all_models(image_path, prediction_dict):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # convert color space from BGR to RGB

    # Ground truth boxes
    for box in prediction_dict["Ground_Truth_boxes"]:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(
            image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1
        )  # Red color

    # Default model boxes
    for box in prediction_dict["Default_YoloNas_boxes"]:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(
            image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1
        )  # Blue color

    # Finetuned model boxes
    for box in prediction_dict["Finetuned_YoloNas_boxes"]:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(
            image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1
        )  # Green color

    # Add labels on top right corner
    cv2.putText(
        image,
        "Ground Truth",
        (image.shape[1] - 200, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        image,
        "Default Model",
        (image.shape[1] - 200, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        image,
        "Finetuned Model",
        (image.shape[1] - 200, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    # Show image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def get_boxex_for_all_models(
    image_path, image_label, default_model, finetuned_model, conf=0.6
):
    # renaming image
    # Split the string at the first occurrence of "_jpg"
    split_name_0 = image_path.split("img_", 1)
    # The split method returns a list, so you need to get the first element
    image_name_0 = "img_" + split_name_0[1]
    # Split the string at the first occurrence of "_jpg"
    split_name = image_name_0.split("_jpg", 1)
    # The split method returns a list, so you need to get the first element
    image_name = split_name[0]

    # get predictions
    # in case of ground truth (no model prediction)
    gt_image = Image.open(image_path)
    # in case of default yolo_nas
    predicted_image_default = default_model.predict(image_path, conf)
    # in case of finetned yolo_nas
    predicted_image_finetuned = finetuned_model.predict(image_path, conf)

    # get bounding boxes
    # in case of ground truth (no model prediction)
    gt_boxes = cxcyxy_to_xyxy(image_label, image_path)
    # in case of default yolo_nas
    default_model_boxes = get_boxes_xyxy(default_model, image_path, conf)
    # in case of finetned yolo_nas
    finetuned_model_boxes = get_boxes_xyxy(finetuned_model, image_path, conf)

    prediction_dict = {}

    prediction_dict["image"] = image_name
    prediction_dict["Ground_Truth_boxes"] = gt_boxes
    prediction_dict["Default_YoloNas_boxes"] = default_model_boxes
    prediction_dict["Finetuned_YoloNas_boxes"] = finetuned_model_boxes

    return prediction_dict


'''
def compute_iou_for_all_models(prediction_dict):
    image_name = prediction_dict["image"]
    gt_boxes = prediction_dict["Ground_Truth_boxes"]
    defalt_yoloNas_boxes = prediction_dict["Default_YoloNas_boxes"]
    finetuned_yoloNas_boxes = prediction_dict["Finetuned_YoloNas_boxes"]

    final_iou_default = []  # only keep highest IoU for each ground truth box
    final_pbb_default = (
        []
    )  # only keep the bb with the highest IoU for each ground truth box

    final_iou_finetuned = []  # only keep highest IoU for each ground truth box
    final_pbb_finetuned = (
        []
    )  # only keep the bb with the highest IoU for each ground truth box

    for gt_box in gt_boxes:
        ls_iou_default = []
        for pred_box in defalt_yoloNas_boxes:
            iou = bbox_iou(gt_box, pred_box)
            ls_iou_default.append(iou)
        idx_max = np.argmax(ls_iou_default)  # find the position with the highest IoU
        final_iou_default.append(ls_iou_default[idx_max])
        final_pbb_default.append(defalt_yoloNas_boxes[idx_max])

    for gt_box in gt_boxes:
        ls_iou_finetuned = []
        for pred_box in finetuned_yoloNas_boxes:
            iou = bbox_iou(gt_box, pred_box)
            ls_iou_finetuned.append(iou)
        idx_max = np.argmax(ls_iou_finetuned)  # find the position with the highest IoU
        final_iou_finetuned.append(ls_iou_finetuned[idx_max])
        final_pbb_finetuned.append(finetuned_yoloNas_boxes[idx_max])

    iou_dict = {}
    iou_dict["image"] = image_name
    iou_dict["IOU_with_default_YoloNAS"] = final_iou_default
    iou_dict["IOU_with_finetuned_YoloNAS"] = final_iou_finetuned

    return iou_dict
'''


def compute_iou_for_all_models(prediction_dict):
    image_name = prediction_dict["image"]
    gt_boxes = prediction_dict["Ground_Truth_boxes"]
    defalt_yoloNas_boxes = prediction_dict["Default_YoloNas_boxes"]
    finetuned_yoloNas_boxes = prediction_dict["Finetuned_YoloNas_boxes"]

    final_iou_default = []
    final_pbb_default = []

    final_iou_finetuned = []
    final_pbb_finetuned = []

    for gt_box in gt_boxes:
        ls_iou_default = []
        for pred_box in defalt_yoloNas_boxes:
            iou = bbox_iou(gt_box, pred_box)
            ls_iou_default.append(iou)

        if ls_iou_default:  # Check if the list is not empty
            # find the position with the highest IoU
            idx_max = np.argmax(ls_iou_default)
            final_iou_default.append(ls_iou_default[idx_max])
            final_pbb_default.append(defalt_yoloNas_boxes[idx_max])
        else:
            # Handle the case where there's no IoU computed
            # Depending on your use-case, you can append a default value or skip
            # For this example, I'll append a default value of -1 to indicate no IoU was computed
            final_iou_default.append(-1)
            final_pbb_default.append([])

    for gt_box in gt_boxes:
        ls_iou_finetuned = []
        for pred_box in finetuned_yoloNas_boxes:
            iou = bbox_iou(gt_box, pred_box)
            ls_iou_finetuned.append(iou)

        if ls_iou_finetuned:
            # find the position with the highest IoU
            idx_max = np.argmax(ls_iou_finetuned)
            final_iou_finetuned.append(ls_iou_finetuned[idx_max])
            final_pbb_finetuned.append(finetuned_yoloNas_boxes[idx_max])
        else:
            final_iou_finetuned.append(-1)
            final_pbb_finetuned.append([])

    iou_dict = {}
    iou_dict["image"] = image_name
    iou_dict["IOU_with_default_YoloNAS"] = final_iou_default
    iou_dict["IOU_with_finetuned_YoloNAS"] = final_iou_finetuned

    return iou_dict
