from webbrowser import get
from super_gradients.training import Trainer
from super_gradients.training.models import get as get_model
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
from data.splitter.yolo import YOLO_Splitter
import torch
import yaml
import json
import os
from ultralytics import NAS
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Niche_YOLO_NAS:
    def __init__(self, path_model, dir_train, dir_val, dir_test, name_task):
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

    def load(self, path_model=None):
        # self.model = YOLO(path_model)
        if path_model is None:
            self.model = NAS("yolo_nas_l")  # NAS(path_model)
        else:
            self.model = get_model(
                "yolo_nas_l", num_classes=80, checkpoint_path=path_model
            )
        print("model %s loaded" % path_model)

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

        # TODO: add test data
        
        self.test_data = coco_detection_yolo_format_val(
            dataset_params={
                "data_dir": os.path.dirname(path_val_txt),
                "images_dir": "images",
                "labels_dir": "labels",
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
            "max_epochs": 2,  ##
            "mixed_precision": False,
            "loss": PPYoloELoss(
                use_static_assigner=False, num_classes=num_classes, reg_max=16
            ),
            "valid_metrics_list": [
                DetectionMetrics_050(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=num_classes,
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=0.01,
                        nms_top_k=1000,
                        max_predictions=300,
                        nms_threshold=0.7,
                    ),
                )
            ],
            "metric_to_watch": "mAP@0.50",
        }

        self.trainer.train(
            model=self.model,
            training_params=train_params,
            train_loader=self.train_data,
            valid_loader=self.val_data,
        )

        best_model_path = os.path.join(self.dir_train, self.name_task, "ckpt_best.pth")
        self.load(best_model_path)
        ## TODO: use get function to load the best model 
        # reference: https://docs.deci.ai/super-gradients/documentation/source/QuickstartBasicToolkit.html
        #self.model = get_model(////
        self.model = get_model("yolo_nas_l", num_classes=num_classes, checkpoint_path=best_model_path)
        
        #########################################

    def get_dataloader(self, path_yaml, data_path_txt, batch_size=16):
        with open(path_yaml, "r") as f:
            yaml_content = yaml.safe_load(f)
        num_classes = yaml_content["nc"]
        # print('num_classes', num_classes)
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
            dataloader_params={"batch_size": batch_size, "num_workers": 2},
        )
        return data
    
    def evaluate_trained_model(self, best_model, dataset_params):
        """
        Evaluates a trained model on test data.

        Parameters:
        - best_model: The trained model to be evaluated.
        - dataset_params: Dictionary containing dataset parameters (like 'classes').

        Returns:
        - The results of the test evaluation.
        """

        test_metrics_list = DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )

        return self.trainer.test(
            model=best_model,
            # Assuming you want to use the test_data from the class instance
            test_loader=self.test_data,
            test_metrics_list=test_metrics_list
        )


# Usage:
# results = evaluate_trained_model(trainer_instance, best_model, test_data, dataset_params)

    def evaluate(self, log_file_txt):
        # TODO: add test data
        # load the best model and evaluate the model on the test split
        # use trainer.test() function to eavalute the model on the split
        #self.trainer.get(///
        self.trainer.test(
            model=best_model,
            test_loader=test_data,
            test_metrics_list=DetectionMetrics_050(
            core_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7)))

        #metrics = self.trainer.test(model=self.model, test_loader=self.test_data_loder)
        #results = self.trainer.test(model=self.model, test_loader=self.test_data_loder)

        # load the log file
        with open(log_file_txt, "r") as f:
            log_lines = f.readlines()
        metrics = []
        for line in log_lines:
            metric = extract_metrics(line)
            if metric:
                metrics.append(metric)
        return metrics

    def evaluation_plot(self, log_file_txt):
        
        df = pd.DataFrame(self.evaluate(log_file_txt))

        # plot loss
        plt.figure(figsize=(10, 6))
        # plot loss
        plt.plot(df['epoch'], df['train_loss'], label="Train Loss")
        plt.plot(df['epoch'], df['valid_loss'], label="Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc="best")
        plt.title('Train and Validation Loss')
        plt.show()


        # plot precision, recall, mAP, F1
        plt.figure(figsize=(10, 6))
        plt.plot(df["epoch"], df["precision@0.50"], label="precision@0.50")
        plt.plot(df["epoch"], df["recall@0.50"], label="recall@0.50")
        plt.plot(df["epoch"], df["mAP@0.50"], label="mAP@0.50")
        plt.plot(df["epoch"], df["F1@0.50"], label="F1@0.50")
        plt.title("Metrics over epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
        
    def forward(self, x):
            return self.model(x)  # Forward pass should use the internal model

    # Ensure that other calls like eval() and train() go to the internal model:
    def eval(self):
        return self.model.eval()

    def train(self, mode=True):
        return self.model.train(mode)


def extract_metrics(log_line):
    regex_pattern = r"Epoch (\d+) \(\d+\/\d+\)\s+-.*Train_PPYoloELoss/loss: (.*?)\s+.*Valid_PPYoloELoss/loss: (.*?)\s+Valid_Precision@0.50: (.*?)\s+Valid_Recall@0.50: (.*?)\s+Valid_mAP@0.50: (.*?)\s+Valid_F1@0.50: (.*?)\s+"

    match = re.search(regex_pattern, log_line)
    if match:
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        valid_loss = float(match.group(3))
        precision = float(match.group(4))
        recall = float(match.group(5))
        mAP50 = float(match.group(6))
        F1 = float(match.group(7))

        return {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "precision@0.50": precision,
            "recall@0.50": recall,
            "mAP@0.50": mAP50,
            "F1@0.50": F1,
        }
    return None

