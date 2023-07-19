from super_gradients.training import Trainer
from super_gradients.training.models import get as get_model
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
import torch
import yaml
import os
from ultralytics import NAS

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"


class Niche_YOLO_NAS:
    def __init__(self, path_model, dir_train, dir_val, name_task):
        self.name_task = name_task
        self.path_model = path_model
        self.dir_train = dir_train
        self.dir_val = dir_val
        self.model = get_model(path_model, pretrained_weights="coco").to(DEVICE)
        self.trainer = Trainer(experiment_name=name_task)
    def load(self, path_model):
        #self.model = YOLO(path_model)
        self.model = NAS('yolo_nas_l')#NAS(path_model)
        print("model %s loaded" % path_model)

    def train(self, path_yaml, path_train_txt, path_val_txt, batch_size, num_epochs):
        with open(path_yaml, 'r') as f:
            yaml_content = yaml.safe_load(f)
        num_classes = yaml_content['nc']

        train_data = coco_detection_yolo_format_train(
            dataset_params={
                'data_dir': os.path.dirname(path_train_txt),
                'images_dir': 'images', #os.path.join(os.path.split(path_train_txt)[0],'images'),
                'labels_dir': 'labels', #os.path.join(os.path.split(path_train_txt)[0],'labels'),
                #'classes': num_classes
                'classes': list(range(num_classes))
            },
            dataloader_params={
                'batch_size': batch_size,   
                'num_workers': 2
            }
        )

        val_data = coco_detection_yolo_format_val(
            dataset_params={
                'data_dir': os.path.dirname(path_val_txt),
                'images_dir': 'images', #os.path.join(os.path.split(path_val_txt)[0],'images'),
                'labels_dir': 'labels', #os.path.join(os.path.split(path_val_txt)[0], 'labels'),
                #'classes': num_classes
                'classes': list(range(num_classes))
            },
            dataloader_params={
                'batch_size': batch_size,
                'num_workers': 2
            }
        )

        train_params = {
            'silent_mode': False,
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
            "max_epochs": 2,
            "mixed_precision": False,
            "loss": PPYoloELoss(
                use_static_assigner=False,
                num_classes=num_classes,
                reg_max=16
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
                        nms_threshold=0.7
                    )
                )
            ],
            "metric_to_watch": 'mAP@0.50'
        }

        self.trainer.train(model=self.model,
                           training_params=train_params,
                           train_loader=train_data,
                           valid_loader=val_data)

    def evaluate(self):
        
        pass
