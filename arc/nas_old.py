from webbrowser import get
from super_gradients.training import Trainer, models
from super_gradients.training.models import get as get_model
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics_050_095
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
from onemetric.cv.object_detection import ConfusionMatrix
import supervision as sv
from onemetric.cv.object_detection import MeanAveragePrecision
from data.splitter.yolo import YOLO_Splitter
import torch
import yaml
import json
import os
import numpy as np
from ultralytics import NAS
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




    

class Niche_YOLO_NAS:
    def __init__(self, model_type, path_model, dir_train, dir_val, dir_test, name_task):
        super(Niche_YOLO_NAS, self).__init__()
        self.name_task = name_task
        self.path_model = path_model
        self.dir_train = dir_train
        self.dir_val = dir_val
        self.dir_test = dir_test
        self.model = get_model(path_model, pretrained_weights="coco").to(DEVICE)
        self.trainer = Trainer(experiment_name=name_task)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.model_type = model_type
        
    '''
    def load(self, path_model=None):
        # self.model = YOLO(path_model)
        if path_model is None:
            self.model = NAS("yolo_nas_l")  # NAS(path_model)
        elif path_model == "yolo_nas_m":
            self.model = models.get(
                "yolo_nas_m", num_classes=80, checkpoint_path=path_model
            )
        elif path_model == "yolo_nas_s":
            self.model = models.get(
                "yolo_nas_s", num_classes=80, checkpoint_path=path_model
            )
        else:
            self.model = models.get(
                "yolo_nas_l", num_classes=80, checkpoint_path=path_model
            )
        print("model %s loaded" % path_model)
        return self.model
    '''
    def train(self, path_yaml, path_train_txt, path_val_txt, batch_size, num_epochs):
        with open(path_yaml, "r") as f:
            yaml_content = yaml.safe_load(f)
        num_classes = yaml_content["nc"]

        self.train_data = coco_detection_yolo_format_train(
            dataset_params={
                "data_dir": os.path.dirname(path_train_txt),
                "images_dir": "images",  # os.path.join(os.path.split(path_train_txt)[0],'images'),
                "labels_dir": "labels",  # os.path.join(os.path.split(path_train_txt)[0],'labels'),
                #'classes': num_classes
                "classes": list(range(num_classes)),
            },
            dataloader_params={"batch_size": batch_size, "num_workers": 2},
        )

        self.val_data = coco_detection_yolo_format_val(
            dataset_params={
                "data_dir": os.path.dirname(path_val_txt),
                "images_dir": "images",  # os.path.join(os.path.split(path_val_txt)[0],'images'),
                "labels_dir": "labels",  # os.path.join(os.path.split(path_val_txt)[0], 'labels'),
                #'classes': num_classes
                "classes": list(range(num_classes)),
            },
            dataloader_params={"batch_size": batch_size, "num_workers": 2},
        )
        ##############################

        train_params = {
            "silent_mode": False,
            "average_best_models": True,
            "warmup_mode": "linear_epoch_step",
            "warmup_initial_lr": 1e-6,
            "lr_warmup_epochs": 3,
            "initial_lr": 5e-4,
            "lr_mode": "cosine",
            "cosine_final_lr_ratio": 0.1,
            "optimizer": "Adam",
            "optimizer_params": {"weight_decay": 0.0001},
            "zero_weight_decay_on_bias_and_bn": True,
            "ema": True,
            "ema_params": {"decay": 0.9, "decay_type": "threshold"},
            "max_epochs": num_epochs,  ##
            "mixed_precision": False,
            "loss": PPYoloELoss(
                use_static_assigner=False, num_classes=num_classes, reg_max=16
            ),
            "valid_metrics_list": [DetectionMetrics_050(score_thres=0.1,
                                                        top_k_predictions=300,
                                                        num_cls=num_classes,
                                                        normalize_targets=True,
                                                        post_prediction_callback=PPYoloEPostPredictionCallback\
                                                            (score_threshold=0.01,nms_top_k=1000,
                                                             max_predictions=300,nms_threshold=0.7,)),
                                   
                                 ],
            "metric_to_watch": "mAP@0.50",
        }

        self.trainer.train(
            model=self.model,
            training_params=train_params,
            train_loader=self.train_data,
            valid_loader=self.val_data,
        )

    
    '''
    def evaluate_trained_model(self, best_model, data_yaml_path, data_type="test"):
        """
        Evaluates a trained model on test data.

        Parameters:
        - best_model: The trained model to be evaluated.
        - dataset_params: Dictionary containing dataset parameters (like 'classes').

        Returns:
        - The results of the test evaluation.
        """
        with open(data_yaml_path, "r") as f:
            yaml_content = yaml.safe_load(f)
        num_classes = yaml_content["nc"]
        # print('num_classes', num_classes)
        if data_type == "test":
            data_path_txt = self.dir_test
            data = coco_detection_yolo_format_val(
                dataset_params={
                    "data_dir": os.path.dirname(data_path_txt),
                    "images_dir": "images",
                    "labels_dir": "labels",
                    "classes": list(range(num_classes)),
                },
                dataloader_params={"batch_size": 16, "num_workers": 2},
            )
        elif data_type == "train":
            data_path_txt = self.dir_train
            data = coco_detection_yolo_format_train(
                dataset_params={
                    "data_dir": os.path.dirname(data_path_txt),
                    # os.path.join(os.path.split(path_train_txt)[0],'images'),
                    "images_dir": "images",
                    # os.path.join(os.path.split(path_train_txt)[0],'labels'),
                    "labels_dir": "labels",
                    #'classes': num_classes
                    "classes": list(range(num_classes)),
                },
                dataloader_params={"batch_size": 16, "num_workers": 2},
            )
        elif data_type == "val":
            data_path_txt = self.dir_val
            data = coco_detection_yolo_format_val(
                dataset_params={
                    "data_dir": os.path.dirname(data_path_txt),
                    "images_dir": "images",
                    "labels_dir": "labels",
                    "classes": list(range(num_classes)),
                },
                dataloader_params={"batch_size": 16, "num_workers": 2},
            )
        else:
            print("data_type is not valid")

        test_metrics_list = [DetectionMetrics_050(
            score_thres=0.5,
            top_k_predictions=300,
            num_cls=len(list(range(num_classes))),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.5,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.5,
            ),
        ),
            DetectionMetrics_050_095(
                score_thres=0.5,
                top_k_predictions=300,
                num_cls=len(list(range(num_classes))),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.5,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.5,
                ),
            ),
        ]

        return self.trainer.test(
            model=best_model,
            test_loader=data,
            test_metrics_list=test_metrics_list,
        )
     '''
        
    def get_evaluation_matrix(
        self, best_model, data_yaml_path, data_type="test", conf=0.5, plot=True
    ):
        if data_type == "test":
            data_dir = self.dir_test
        elif data_type == "train":
            data_dir = self.dir_train
        elif data_type == "val":
            data_dir = self.dir_val
        ds = sv.DetectionDataset.from_yolo(
            images_directory_path=data_dir + "/images",
            annotations_directory_path=data_dir + "/labels",
            data_yaml_path=data_yaml_path,
            force_masks=False,
        )

        predictions = {}
        for image_name, image in ds.images.items():
            result = list(best_model.predict(image, conf=conf))[0]
            detections = sv.Detections(
                xyxy=result.prediction.bboxes_xyxy,
                confidence=result.prediction.confidence,
                class_id=result.prediction.labels.astype(int),
            )
            predictions[image_name] = detections

        keys = list(ds.images.keys())
        annotation_batches, prediction_batches = [], []

        for key in keys:
            annotation = ds.annotations[key]
            annotation_batch = np.column_stack((annotation.xyxy, annotation.class_id))
            annotation_batches.append(annotation_batch)

            prediction = predictions[key]
            prediction_batch = np.column_stack(
                (prediction.xyxy, prediction.class_id, prediction.confidence)
            )
            prediction_batches.append(prediction_batch)

        confusion_matrix = ConfusionMatrix.from_detections(
            true_batches=annotation_batches,
            detection_batches=prediction_batches,
            num_classes=len(ds.classes),
            conf_threshold=conf,
        )
        print("Confusion Matrix:", confusion_matrix.matrix)

        

        if plot:
            confusion_matrix.plot(
                os.path.join(data_dir, "confusion_matrix.png"), class_names=ds.classes
            )
        else:
            pass

        return confusion_matrix     
    
    
    
    def evaluate_trained_model(self, model_type, data_yaml_path, data_type="test"):
        """
        Evaluates a trained model on test data.

        Parameters:
        - model_type: Type of the model ('yolo_nas_l', 'yolo_nas_m', 'yolo_nas_s').
        - data_yaml_path: Path to the YAML file containing dataset parameters.
        - data_type: Type of the data to evaluate on ('test', 'train', 'val').

        Returns:
        - The results of the test evaluation.
        """
        # Load the model based on the model type
        best_model = self.load(model_type)

        with open(data_yaml_path, "r") as f:
            yaml_content = yaml.safe_load(f)
        num_classes = yaml_content["nc"]

        # Data loading based on data_type
        if data_type == "test":
            data_path_txt = self.dir_test
            data = coco_detection_yolo_format_val(
                dataset_params={
                    "data_dir": os.path.dirname(data_path_txt),
                    "images_dir": "images",
                    "labels_dir": "labels",
                    "classes": list(range(num_classes)),
                },
                dataloader_params={"batch_size": 16, "num_workers": 2},
            )
        elif data_type == "train":
            data_path_txt = self.dir_train
            data = coco_detection_yolo_format_train(
                dataset_params={
                    "data_dir": os.path.dirname(data_path_txt),
                    "images_dir": "images",
                    "labels_dir": "labels",
                    "classes": list(range(num_classes)),
                },
                dataloader_params={"batch_size": 16, "num_workers": 2},
            )
        elif data_type == "val":
            data_path_txt = self.dir_val
            data = coco_detection_yolo_format_val(
                dataset_params={
                    "data_dir": os.path.dirname(data_path_txt),
                    "images_dir": "images",
                    "labels_dir": "labels",
                    "classes": list(range(num_classes)),
                },
                dataloader_params={"batch_size": 16, "num_workers": 2},
            )
        else:
            raise ValueError("data_type is not valid")

        # Setup test metrics list
        test_metrics_list = [DetectionMetrics_050(
            score_thres=0.5,
            top_k_predictions=300,
            num_cls=len(list(range(num_classes))),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.5,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.5,
            ),
        ),
        DetectionMetrics_050_095(
            score_thres=0.5,
            top_k_predictions=300,
            num_cls=len(list(range(num_classes))),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.5,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.5,
            ),
        )]

        return self.trainer.test(
            model=best_model,
            test_loader=data,
            test_metrics_list=test_metrics_list,
        )

    def load(self,checkpoint_path,model_type, path_model=None):
        """
    Load the model based on the specified type and checkpoint path.

    Parameters:
    - model_type: A string indicating the model type ('yolo_nas_l', 'yolo_nas_m', 'yolo_nas_s').
    - checkpoint_path: Path to the saved checkpoint. If None, loads model without pretrained weights.

    Returns:
    - Loaded model.
    """
    # Specify the number of classes if needed
    num_classes = 80  # Adjust this based on your dataset

    if model_type == "yolo_nas_l":
        self.model = get_model("yolo_nas_l", num_classes=num_classes)
    elif model_type == "yolo_nas_m":
        self.model = get_model("yolo_nas_m", num_classes=num_classes)
    elif model_type == "yolo_nas_s":
        self.model = get_model("yolo_nas_s", num_classes=num_classes)
    else:
        raise ValueError("Unknown model type: {}".format(model_type))

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except KeyError:
            # If the state_dict is directly in the checkpoint
            self.model.load_state_dict(checkpoint)
        except RuntimeError as e:
            print("RuntimeError while loading the model:", e)
            print("Attempting to load model with non-strict state dict...")
            self.model.load_state_dict(checkpoint, strict=False)

    self.model.to(DEVICE)
    print("model {} loaded".format(model_type))
    return self.model
    
    
    


    