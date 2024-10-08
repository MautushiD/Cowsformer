"""
A class to organize YOLO-structured data

Methods
---

get_images
    list of absolute paths of images in root/<split>/images
clone
    copy root/train and root/test to root/<folder_name>
shuffle_train_val
    shuffle self.ls_train_images and assign to
save_yaml
save_txt


Folder structure
---
root/
    train/ (required)
        images/
            img_1_13_jpg.rf.d69528304f2d10b633c6d94982185cb2.jpg
            img_2_13_jpg.rf.d69528304f2d10b633c6d94982185cb2.jpg
            ...
        labels/
            img_1_13_jpg.rf.d69528304f2d10b633c6d94982185cb2.txt
            img_2_13_jpg.rf.d69528304f2d10b633c6d94982185cb2.txt
            ...
    test/ (required)
        images/
            img_3.jpg
            img_4.jpg
            ...
        labels/
            img_3.txt
            img_4.txt

    test.txt (generated)
    train.txt (generated)
    val.txt (generated)
    data.yaml (generated)

Example YAML
---
path: /home/niche/cowsformer/data/cow200/yolov5/run3
train: "train.txt"
val: "val.txt"
test: "test.txt"
names:
  0: none
  1: cow

Example train.txt
---

/home/niche/cowsformer/data/cow200/yolov5/run0/images/img_32_jpg
/home/niche/cowsformer/data/cow200/yolov5/run0/images/img_1_jpg
/home/niche/cowsformer/data/cow200/yolov5/run0/images/img_1_26_jpg
/home/niche/cowsformer/data/cow200/yolov5/run0/images/img_1_62_jpg
/home/niche/cowsformer/data/cow200/yolov5/run0/images/img_1_10_jpg
/home/niche/cowsformer/data/cow200/yolov5/run0/images/img_3_11_jpg
/home/niche/cowsformer/data/cow200/yolov5/run0/images/img_4_3_jpg
"""

import os
import shutil
import random
import PIL
import torch
import glob

import numpy as np
import supervision as sv

# local imports
from bbox import xywh2xyxy


class YOLO_API:
    def __init__(
        self,
        root: str,
    ):
        self.root = root
        self.ls_train_images_all = self.get_images("train")
        self.ls_train_images = None
        self.ls_val_iamges = None
        self.ls_test_images = self.get_images("test")
        self.save_txt("test")

    def get_PIL(self, split, idx):
        """
        get PIL image from split and idx

        params
        ------
        split: str
            "train" or "test"
        idx: int
            index of the image

        return
        ------
        PIL.Image
        """
        if split == "train":
            path = self.ls_train_images_all[idx]
        else:
            path = self.ls_test_images[idx]
        return PIL.Image.open(path)

    def get_images(self, split):
        """
        search images names (.jpg) in root/<split>/images

        params
        ------
        split: str
            "train" or "test"

        return
        ------
        a list of aboslute paths of images
        """
        return self.get_filepaths(split, "images")

    def get_labels(self, split):
        """
        search label names (.txt) in root/<split>/labels

        params
        ------
        split: str
            "train" or "test"

        return
        ------
        a list of aboslute paths of labels
        """
        return self.get_filepaths(split, "labels")
    
    def get_labels_pred(self, split):
        """
        Dynamically find the directory name ending with 'labelsPred' and fetch label files from it.

        Parameters
        ----------
        split : str
            "train" or "test"

        Returns
        -------
        list
            A list of absolute paths of labels
        """
        base_path = os.path.join(self.root, split)
        labels_pred_dir_pattern = os.path.join(base_path, "*labelsPred")
        labels_pred_dirs = glob.glob(labels_pred_dir_pattern)
        
        # Assuming there's only one such directory
        if labels_pred_dirs:
            # Extract the dynamic folder name part that matches the criteria
            dynamic_folder_name = os.path.basename(labels_pred_dirs[0])
            # Now pass this dynamic folder name to self.get_filepaths
            return self.get_filepaths(split, dynamic_folder_name)
        else:
            return []  # Return an empty list if no matching directory is found


    def get_filepaths(self, split, folder):
        """
        get file paths of images or labels

        params
        ---
        split: str
            "train" or "test"
        folder: str
            "images" or "labels"
        """
        path_files = os.path.join(self.root, split, folder)
        ext = ".jpg" if folder == "images" else ".txt"
        ls_files = [f for f in os.listdir(path_files) if f.endswith(ext)]
        ls_files = [os.path.join(path_files, f) for f in ls_files]
        return sorted(ls_files)


    
    def get_gt_pred_detections(self, gt_img_path,gt_label_path,pred_label_path):
            
        gt_detections = []
        gt_labels = [f for f in os.listdir(gt_label_path) if f.endswith(".txt")]
        gt_labels = sorted([os.path.join(gt_label_path, f) for f in gt_labels])
        gt_images = self.get_images(gt_img_path)
        n_samples_gt = len(gt_images)
        
        pred_detections = []
        pred_labels = [f for f in os.listdir(pred_label_path) if f.endswith(".txt")]
        pred_labels = sorted([os.path.join(pred_label_path, f) for f in pred_labels])
        n_samples_pred = len(pred_labels)
        
        if n_samples_gt != n_samples_pred:
            print('gt_images and pred images are not same')
        else:
            pass
        
        
        for i in range(n_samples_gt):
            # get image info
            image = PIL.Image.open(gt_images[i])
            img_w, img_h = image.size
            # get annotation
            gt_label = gt_labels[i]
            pred_label = pred_labels[i]
            
            with open(gt_label, "r") as f:
                gt_lines = f.readlines()
                gt_lines = [i.strip() for i in gt_lines]
                
            with open(pred_label, "r") as f:
                pred_lines = f.readlines()
                pred_lines = [i.strip() for i in pred_lines]
                
            if not gt_lines:
                #print('found empty GT')
                continue
            if gt_lines and not pred_lines:
                #print('case found: empty pred')
                pred_lines = ["0 0.000001 0.000002 0.000003 0.000004 0.0000001"]

            ### get gt detections
            
            gt_ls_xyxy = []
            gt_ls_cls = []
            gt_ls_conf = []
            for i in gt_lines:
                gt_parts = i.split(" ")
                gt_class_id = int(gt_parts[0])
                gt_coords = tuple(
                    map(float, gt_parts[1:5])
                )  # x_center, y_center, width, height
                #conf = float(parts[5]) if path_preds else None
                
                gt_xyxy = xywh2xyxy(
                    gt_coords,
                    img_size=(img_w, img_h),
                )
                # append to lists
                gt_ls_xyxy.append(gt_xyxy)
                gt_ls_cls.append(gt_class_id)
                
            # create sv.Detections
            gt_ls_xyxy = torch.stack(gt_ls_xyxy).numpy()
            gt_ls_cls = np.array(gt_ls_cls)
            gt_ls_conf = np.array(gt_ls_conf)
            gt_detection = sv.Detections(
                gt_ls_xyxy,
                class_id=gt_ls_cls,
                confidence= None,
            )
            gt_detections.append(gt_detection)
            
            ### get pred detections
            
            pred_ls_xyxy = []
            pred_ls_cls = []
            pred_ls_conf = []
            for i in pred_lines:
                pred_parts = i.split(" ")
                #print(parts)
                pred_class_id = int(pred_parts[0])
                pred_coords = tuple(
                    map(float, pred_parts[1:5])
                )  # x_center, y_center, width, height
                pred_conf = float(pred_parts[5]) 
                pred_coords = tuple(value for value in pred_coords)

                pred_xyxy = torch.tensor(pred_coords)
                # append to lists
                pred_ls_xyxy.append(pred_xyxy)
                pred_ls_cls.append(pred_class_id)
                pred_ls_conf.append(pred_conf)
            # create sv.Detections
            pred_ls_xyxy = torch.stack(pred_ls_xyxy).numpy()
            pred_ls_cls = np.array(pred_ls_cls)
            pred_ls_conf_array = np.array(pred_ls_conf)
            
            pred_detection = sv.Detections(
                pred_ls_xyxy,
                class_id=pred_ls_cls,
                confidence=pred_ls_conf_array,
            )
            pred_detections.append(pred_detection)
            
            
        return gt_detections,pred_detections


    def clone(self, folder_name):
        """
        copy root/train and root/test to
        root/<folder_name>/train and root/<folder_name>/test
        """
        path_train = os.path.join(self.root, "train")
        path_test = os.path.join(self.root, "test")
        path_folder = os.path.join(self.root, folder_name)
        if os.path.exists(path_folder):
            shutil.rmtree(path_folder)
        os.mkdir(path_folder)
        shutil.copytree(path_train, os.path.join(path_folder, "train"))
        shutil.copytree(path_test, os.path.join(path_folder, "test"))
        # copy yaml and other txt
        shutil.copy(os.path.join(self.root, "data.yaml"), path_folder)
        shutil.copy(os.path.join(self.root, "train.txt"), path_folder)
        shutil.copy(os.path.join(self.root, "test.txt"), path_folder)

    def shuffle_train_val(self, n=None, k=5):
        """
        shuffle self.ls_train_images and assign to
        self.ls_train_images and self.ls_val_images

        params
        ------
        n: None or int or float
            None: use all images
            int: number of images to be included in the train/val set
            float: ratio of images to be included in the train/val set
        k: int
            how many folds to split the train/val set
        """
        # determine n
        total_n = len(self.ls_train_images_all)
        if n is None:
            n = total_n
        elif isinstance(n, float):
            n = int(n * total_n)
        n_val = int(n / k)
        # shuffle training images
        random.shuffle(self.ls_train_images_all)
        train_images = self.ls_train_images_all[:n]
        # split train/val
        self.ls_train_images = train_images[:-n_val]

        self.ls_val_images = train_images[-n_val:]
        self.save_txt("train")
        self.save_txt("val")

    def save_yaml(self, classes, name="data.yaml"):
        """
        make data.yaml in root

        params
        ------
        classes: list
            e.g., ["cow", "none"]

        name: str
            name of the yaml file

        """
        path_yaml = os.path.join(self.root, name)
        with open(path_yaml, "w") as f:
            f.write(f"path: {self.root}\n")
            f.write(f'train: "train.txt"\n')
            f.write(f'val: "val.txt"\n')
            f.write(f'test: "test.txt"\n')
            f.write("names:\n")
            for i, c in enumerate(classes):
                f.write(f"  {i}: {c}\n")

    def save_txt(self, split):
        """
        save <split>.txt in root
        """
        path_txt = os.path.join(self.root, f"{split}.txt")
        with open(path_txt, "w") as f:
            for img in getattr(self, f"ls_{split}_images"):
                f.write(img + "\n")

    def path_yaml(self):
        return os.path.join(self.root, "data.yaml")
