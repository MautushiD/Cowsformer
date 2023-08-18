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
import torch
from torchvision import transforms


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
        class_id, x_center, y_center, width, height = map(float, line.strip().split())

        # Convert to x_min, y_min, x_max, y_max format
        x_min = (x_center - width / 2) * image_width
        y_min = (y_center - height / 2) * image_height
        x_max = (x_center + width / 2) * image_width
        y_max = (y_center + height / 2) * image_height

        boxes.append([x_min, y_min, x_max, y_max])

    return boxes


def get_boxes_xyxy(model, image_path, conf=0.6):
    ### Input:
    # model : yolo_nas_l // best_model
    # image_path: 'pth/to/local/img_1.jpg'
    # conf = confidence

    ### Output:
    # all_boxes = list of list | coordinate of all bounding boxes

    results = model.predict(image_path, conf)
    all_boxes = []
    for image_prediction in results:
        bboxes = image_prediction.prediction.bboxes_xyxy
        for i, (bbox) in enumerate(bboxes):
            all_boxes.append(bbox.tolist())

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
    ### renaming image
    # Split the string at the first occurrence of "_jpg"
    split_name_0 = image_path.split("img_", 1)
    # The split method returns a list, so you need to get the first element
    image_name_0 = "img_" + split_name_0[1]
    # Split the string at the first occurrence of "_jpg"
    split_name = image_name_0.split("_jpg", 1)
    # The split method returns a list, so you need to get the first element
    image_name = split_name[0]

    ### get predictions
    # in case of ground truth (no model prediction)
    gt_image = Image.open(image_path)
    # in case of default yolo_nas
    predicted_image_default = default_model.predict(image_path, conf)
    # in case of finetned yolo_nas
    predicted_image_finetuned = finetuned_model.predict(image_path, conf)

    ### get bounding boxes
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
    
    #default_boxes = []
    #finetuned_boxes = []
    #gt_boxes = []

    # Construct and return the prediction dictionary
    #prediction_dict = {
     #   "Default_YoloNas_boxes": default_boxes,
      #  "Finetuned_YoloNas_boxes": finetuned_boxes,
       # "Ground_Truth_boxes": gt_boxes
    #}

    return prediction_dict




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
    iou_dict["Predicted_boxes_with_finetuned_YoloNAS"] = final_pbb_finetuned
    iou_dict["Predicted_boxes_with_default_YoloNAS"] = final_pbb_default
    
    return iou_dict

################
'''
def parse_label_file(label_path):
    """
    Parses a YOLO format label file.
    Args:
    - label_path: Path to the label file
    Returns:
    - A list of bounding boxes. Each bounding box is represented as a dictionary.
    """
    boxes = []
    with open(label_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            elements = line.strip().split()
            cls, x_center, y_center, width, height = map(float, elements)
            box = {
                'class': cls,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            }
            boxes.append(box)
    return boxes


def compute_AP_for_image(iou_list, iou_threshold=0.5):
    # Sorting the predictions based on confidence might be required,
    # but assuming you've already done this when you call model.predict
    true_positives = np.array(
        [1 if iou >= iou_threshold else 0 for iou in iou_list])
    false_positives = 1 - true_positives

    # Compute the cumulative sums
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)

    recalls = tp_cumsum / float(len(iou_list))
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Interpolation step: For each level of recall, find the max precision:
    interpolated_precisions = []
    for recall_level in np.linspace(0.0, 1.0, 100):
        try:
            args = np.where(recalls >= recall_level)[0]
            interpolated_precisions.append(max(precisions[args]))
        except:
            interpolated_precisions.append(0.0)

    average_precision = np.mean(interpolated_precisions)
    return average_precision


def read_label_file(label_file_path):
    """
    Reads a YOLOv5 label file and returns the ground truths.
    """
    with open(label_file_path, 'r') as file:
        lines = file.readlines()
    # Convert each line into a list [class_id, x_center, y_center, width, height]
    boxes = [list(map(float, line.strip().split())) for line in lines]
    return boxes


def preprocess_image(image_path):
    transform = transforms.Compose([
        # Resize to the input size your model expects. Change if necessary.
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # normalization based on ImageNet stats
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # unsqueeze to add batch dimension

def compute_mAP_for_test_data(model, images):
    # Initialize the data
    det_preds = []
    det_gts = []

    # Label directory
    image_label_dir = "/Users/mautushid/github/Cowsformer/data/cow200/yolov5/test/labels"

    for image in images:
        tensor_image = preprocess_image(image)  # preprocess the image
        with torch.no_grad():
            preds = model(tensor_image)
        det_preds.append(preds)

        # Get the filename (without extension) of the current image
        image_name = os.path.basename(image).rsplit('.', 1)[0]  # Extract the filename without extension
        label_file = os.path.join(image_label_dir, f"{image_name}.txt")

        # read and store the ground truths (based on your specific format)
        gt = parse_label_file(label_file)
        det_gts.append(gt)

    mAP = calculate_mAP(det_preds, det_gts)
    return mAP


def some_mAP_computation_method(det_preds, gt_boxes):
    # Placeholder for actual mAP computation
    # This is a dummy value; in a real scenario, you'd compare predictions with ground truths
    return 0.9
'''
def evaluate_predictions(prediction_dict, iou_threshold=0.5):
    image_name = prediction_dict["image"]
    gt_boxes = prediction_dict["Ground_Truth_boxes"]
    defalt_yoloNas_boxes = prediction_dict["Default_YoloNas_boxes"]
    finetuned_yoloNas_boxes = prediction_dict["Finetuned_YoloNas_boxes"]

    TP_default = 0
    FN_default = 0
    FP_default = len(defalt_yoloNas_boxes)

    TP_finetuned = 0
    FN_finetuned = 0
    FP_finetuned = len(finetuned_yoloNas_boxes)

    for gt_box in gt_boxes:
        # Default Model
        ious_default = [bbox_iou(gt_box, pred_box) for pred_box in defalt_yoloNas_boxes]
        if ious_default and max(ious_default) >= iou_threshold:
            TP_default += 1
            FP_default -= 1  # Remove false positive since this prediction is true
        else:
            FN_default += 1

        # Finetuned Model
        ious_finetuned = [bbox_iou(gt_box, pred_box) for pred_box in finetuned_yoloNas_boxes]
        if ious_finetuned and max(ious_finetuned) >= iou_threshold:
            TP_finetuned += 1
            FP_finetuned -= 1
        else:
            FN_finetuned += 1

    # Calculate precision and recall for both models
    precision_default = TP_default / (TP_default + FP_default) if TP_default + FP_default != 0 else 0
    recall_default = TP_default / (TP_default + FN_default) if TP_default + FN_default != 0 else 0

    precision_finetuned = TP_finetuned / (TP_finetuned + FP_finetuned) if TP_finetuned + FP_finetuned != 0 else 0
    recall_finetuned = TP_finetuned / (TP_finetuned + FN_finetuned) if TP_finetuned + FN_finetuned != 0 else 0

    return {
        "image": image_name,
        "precision_default": precision_default,
        "recall_default": recall_default,
        "precision_finetuned": precision_finetuned,
        "recall_finetuned": recall_finetuned
    }
