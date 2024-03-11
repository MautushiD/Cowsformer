from webbrowser import get
from super_gradients.training import models, Trainer
#from models.nas_trainer import Trainer
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
#from lightning.pytorch.callbacks import ModelCheckpoint
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
    def __init__(self,  path_model, dir_train, dir_val, dir_test, name_task):
        super(Niche_YOLO_NAS, self).__init__()
        self.name_task = name_task
        self.path_model = path_model
        self.dir_train = dir_train
        self.dir_val = dir_val
        self.dir_test = dir_test
        self.model = get_model(self.path_model, pretrained_weights="coco").to(DEVICE)
        self.trainer = Trainer(experiment_name=name_task)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load(self, path_model=None, finetuned_chekpoint_path=None):
        # self.model = YOLO(path_model)
        if path_model is None:
            self.model = NAS("yolo_nas_l")  # NAS(path_model)
        elif path_model == "yolo_nas_m":
            self.model = models.get(
                "yolo_nas_m", num_classes=80, checkpoint_path=finetuned_chekpoint_path
            )
        elif path_model == "yolo_nas_s":
            self.model = models.get(
                "yolo_nas_s", num_classes=80, checkpoint_path=finetuned_chekpoint_path
            )
        else:
            self.model = models.get(
                "yolo_nas_l", num_classes=80, checkpoint_path=finetuned_chekpoint_path
            )
        print("model %s loaded" % path_model)
        return self.model
    ############

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
                "images_dir": "train/images",  # os.path.join(os.path.split(path_train_txt)[0],'images'),
                "labels_dir": "train/labels",  # os.path.join(os.path.split(path_train_txt)[0],'labels'),
                #'classes': num_classes
                "classes": list(range(num_classes)),
            },
            dataloader_params={"batch_size": batch_size, "num_workers": 2},
        )

        self.val_data = coco_detection_yolo_format_val(
            dataset_params={
                "data_dir": os.path.dirname(path_val_txt),
                "images_dir": "val/images",  # os.path.join(os.path.split(path_val_txt)[0],'images'),
                "labels_dir": "val/labels",  # os.path.join(os.path.split(path_val_txt)[0], 'labels'),
                #'classes': num_classes
                "classes": list(range(num_classes)),
            },
            dataloader_params={"batch_size": batch_size, "num_workers": 2},
        )
        ##############################

        train_params = {
            #"resume": False, ### added new
            "ckpt_name": False,
            "silent_mode": False,
            "average_best_models": False,
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
        
    
    def evaluate_test_set(self,ROOT,yolo_base, config, exp_name, n, iteration):
        config_short = config.split("_")[-1]

        dir_train = ROOT + "/data/"+config+"/tv/" + exp_name+"_" + yolo_base + "_" + \
            str(n) + "_" + str(iteration) + "_" + config_short + "_" + \
            yolo_base + "_" + str(n) + "_" + str(iteration) + "/" + "train"
        dir_val = ROOT + "/data/"+config+"/tv/" + exp_name+"_" + yolo_base + "_" + \
            str(n) + "_" + str(iteration) + "_" + config_short + "_" + \
            yolo_base + "_" + str(n) + "_" + str(iteration) + "/" + "val"
        dir_test = ROOT + "/data/"+config + "/test"

        data_yaml_path = ROOT + "/data/"+config+"/tv/" + exp_name+"_" + yolo_base + "_" + \
            str(n) + "_" + str(iteration) + "_" + config_short + "_" + \
            yolo_base + "_" + str(n) + "_" + str(iteration) + "/" + "data.yaml"
        base_dir = ROOT + "/checkpoints/n" + \
            str(n) + "_" + yolo_base + "_i" + \
            str(iteration) + "_" + config_short
        items_under_base = os.listdir(base_dir)[0]
        finetuned_model_path = base_dir + "/" + items_under_base + "/ckpt_best.pth"
        output_dir = dir_test + "/" + exp_name + "_" + yolo_base + "_" + \
            str(n)+"_"+str(iteration) + "_labelsPred"

        # Creating instance of Niche_YOLO_NAS class
        my_nas = Niche_YOLO_NAS(yolo_base, dir_train,
                                dir_val, dir_test, "cow200")
        predictions = my_nas.prediction(data_yaml_path, finetuned_model_path)
        my_nas.write_predictions(predictions, output_dir)

    def remove_ckpt(self, checkpoint_dir,  type = 'latest'):
        """
        Removes the 'ckpt_latest.pth' file from the experiment's checkpoint directory within
        the given base checkpoint directory. It searches for a subdirectory starting with 'RUN'
        and keeps only the 'ckpt_best.pth' file in that subdirectory.

        Args:
            checkpoint_dir (str): Base path to the checkpoint directory which contains
                                subdirectories for each experiment.
        """
        # Find the subdirectory starting with 'RUN'
        run_dir = None
        for item in os.listdir(checkpoint_dir):
            if item.startswith('RUN') and os.path.isdir(os.path.join(checkpoint_dir, item)):
                run_dir = item
                break
        
        # Check if we found a 'RUN' directory
        if run_dir is not None:
            # Construct the full path to the 'ckpt_latest.pth' file
            
            if type == 'latest':
            
                latest_ckpt_path = os.path.join(checkpoint_dir, run_dir, 'ckpt_latest.pth')
            elif type == 'best':
                latest_ckpt_path = os.path.join(
                    checkpoint_dir, run_dir, 'ckpt_best.pth')
            else:
                print('Type must be lateast or best')

            # Check if the file exists
            if os.path.isfile(latest_ckpt_path):
                # Remove the 'ckpt_latest.pth' file
                os.remove(latest_ckpt_path)
                print(f"'{latest_ckpt_path}' has been removed.")
            else:
                print(f"No {type} checkpoint .pth file found to remove in: {os.path.join(checkpoint_dir, run_dir)}")
        else:
            print(f"No 'RUN' directory found in: {checkpoint_dir}")
            
            
    def keep_best_ckpt(checkpoint_dir):
        """
        Removes all files except for the 'ckpt_best.pth' and '.txt' files from the experiment's
        checkpoint directory within the given base checkpoint directory. It searches for a
        subdirectory starting with 'RUN' and performs the cleanup in that subdirectory.

        Args:
            checkpoint_dir (str): Base path to the checkpoint directory which contains
                                subdirectories for each experiment.
        """
        # Find the subdirectory starting with 'RUN'
        run_dir = None
        for item in os.listdir(checkpoint_dir):
            if item.startswith('RUN') and os.path.isdir(os.path.join(checkpoint_dir, item)):
                run_dir = item
                break
        
        # Check if we found a 'RUN' directory
        if run_dir is not None:
            # Get the full path of the 'RUN' directory
            full_run_dir = os.path.join(checkpoint_dir, run_dir)

            # Iterate over the files in the 'RUN' directory
            for file in os.listdir(full_run_dir):
                file_path = os.path.join(full_run_dir, file)
                # Remove all files except 'ckpt_best.pth' and '.txt' files
                if not (file == 'ckpt_best.pth' or file.endswith('.txt')):
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Removed '{file_path}'")
                    elif os.path.isdir(file_path):
                        # If it's a directory, recursively delete it
                        import shutil
                        shutil.rmtree(file_path)
                        print(f"Removed directory '{file_path}'")
            print(f"Cleanup complete in: {full_run_dir}")
        else:
            print(f"No 'RUN' directory found in: {checkpoint_dir}")
                
            

        
    
    
    def prediction(self,data_yaml_path,finetuned_model_path,CONFIDENCE_TRESHOLD = 0.5 ):
        
        ds = sv.DetectionDataset.from_yolo(
        images_directory_path=self.dir_test+"/images",
        annotations_directory_path=self.dir_test+"/labels",
        data_yaml_path=data_yaml_path,
        force_masks=False)
        model = self.load(self.path_model,finetuned_model_path)
        predictions = {}

        for image_name, image in ds.images.items():
            result = (model.predict(
                image, conf=CONFIDENCE_TRESHOLD))
            detections = sv.Detections(
                xyxy=result.prediction.bboxes_xyxy,
                #xyxy=result.prediction.bboxes_xywh,
                confidence=result.prediction.confidence,
                class_id=result.prediction.labels.astype(int)
            )
            predictions[image_name] = detections
            
        return predictions
    def predictionLocal(self,data_yaml_path,finetuned_model_path,CONFIDENCE_TRESHOLD = 0.5 ):
        
        ds = sv.DetectionDataset.from_yolo(
        images_directory_path=self.dir_test+"/images",
        annotations_directory_path=self.dir_test+"/labels",
        data_yaml_path=data_yaml_path,
        force_masks=False)
        model = self.load(self.path_model,finetuned_model_path)
        predictions = {}

        for image_name, image in ds.images.items():
            result = list(model.predict(
                image, conf=CONFIDENCE_TRESHOLD))[0]
            detections = sv.Detections(
                xyxy=result.prediction.bboxes_xyxy,
                #xyxy=result.prediction.bboxes_xywh,
                confidence=result.prediction.confidence,
                class_id=result.prediction.labels.astype(int)
            )
            predictions[image_name] = detections
            
        return predictions
    
    def write_predictions(self,predictions,output_directory):
        # Ensure the output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Iterate over the predictions and write them to files
        for image_path, detections in predictions.items():
            # Extract the base filename without the directory path and extension
            base_filename = os.path.basename(image_path)  # Gets the filename from the path
            base_filename = os.path.splitext(base_filename)[0]  # Removes the file extension
    
            # Prepare the output file path
            output_file_path = os.path.join(output_directory, base_filename + '.txt')

            # Open the file and write the detections
            with open(output_file_path, 'w') as file:
                for bbox, conf, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
                    # Ensure numbers are formatted as strings with appropriate precision
                    bbox_str = ' '.join(f"{x:.2f}" for x in bbox)  # Formats bbox coordinates
                    detection_str = f"{class_id} {bbox_str} {conf:.5f}\n"  # Formats the entire detection string
                    
                    # Write to file
                    file.write(detection_str)
        
        
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
        class_ess = yaml_content["names"]
        #print('num_classes', num_classes)
        if data_type == "test":
            data_path_txt = self.dir_test
            data = coco_detection_yolo_format_val(
                dataset_params={
                    "data_dir": os.path.dirname(data_path_txt),
                    "images_dir": "test/images",
                    "labels_dir": "test/labels",
                    "classes": class_ess,
                },
                dataloader_params={"batch_size": 16, "num_workers": 2},
            )
        elif data_type == "test_old":
            data_path_txt = self.dir_test
            data = coco_detection_yolo_format_val(
                dataset_params={
                    "data_dir": os.path.dirname(data_path_txt),
                    "images_dir": "test_old/images",
                    "labels_dir": "test_old/labels",
                    "classes": class_ess,
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
                    "classes": class_ess,
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
                    "classes": class_ess,
                },
                dataloader_params={"batch_size": 16, "num_workers": 2},
            )
        else:
            print("data_type is not valid")

        test_metrics_list = [DetectionMetrics_050(
            score_thres=0.5,
            top_k_predictions=300,
            num_cls=1,#len(list(range(num_classes))),
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
                num_cls=1,#len(list(range(num_classes))),
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
    
    
 ########################################################################
    '''
    def get_map_scores(self, best_model, data_yaml_path, data_type="test"):
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

        predictions = []
        targets = []
        for image_name, image in ds.images.items():
            result = list(best_model.predict(image))[0]
            detection_batch = np.column_stack(
                (result.prediction.bboxes_xyxy, result.prediction.labels.astype(int), result.prediction.confidence)
            )
            predictions.append(detection_batch)

            annotation = ds.annotations[image_name]
            target_batch = np.column_stack((annotation.xyxy, annotation.class_id))
            targets.append(target_batch)

        mean_average_precision = sv.MeanAveragePrecision.from_tensors(
            predictions=predictions,
            targets=targets,
        )

        map50 = mean_average_precision.map50
        map50_95 = mean_average_precision.map50_95

        return {"mAP@50": map50, "mAP@50:95": map50_95}

    '''  
    def get_map_scores(self, best_model, data_yaml_path, data_type="test"):
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

        predictions = []
        targets = []
        for image_name, image in ds.images.items():
            result = list(best_model.predict(image))[0]
            detection_batch = np.column_stack(
                (result.prediction.bboxes_xyxy, result.prediction.labels.astype(int), result.prediction.confidence)
            )
            predictions.append(detection_batch)

            annotation = ds.annotations[image_name]
            target_batch = np.column_stack((annotation.xyxy, annotation.class_id))
            targets.append(target_batch)

        mean_average_precision = sv.MeanAveragePrecision.from_tensors(
            predictions=predictions,
            targets=targets,
        )

    # Placeholder for calculating additional metrics
    # You need to implement the calculation based on your framework's capabilities or manually
        precision = 0.0  # Implement calculation
        recall = 0.0  # Implement calculation
        f1 = 0.0  # Implement calculation if precision and recall are available
        n_all = len(predictions)  # Total number of predictions
        n_fn = 0  # Implement calculation for false negatives
        n_fp = 0  # Implement calculation for false positives

    # Update the F1 score calculation if precision and recall are non-zero
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "map5095": mean_average_precision.map50_95,
            "map50": mean_average_precision.map50,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_all": int(n_all),
            "n_fn": int(n_fn),
            "n_fp": int(n_fp),
        }   
def get_checkpoint(dir_out):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=dir_out,
        mode="min",
        save_top_k=1,
        verbose=False,
        save_last=False,
        filename="model-{val_loss:.3f}",
    )
    return checkpoint_callback