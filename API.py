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

    def get_gt_detections(self, gt_label_path, gt_img_path):
        """
        get sv.Detections from labels

        params
        ------
        split: str
            "train" or "test"
        path_results: str
            path to the dir of predictions (.txt).
            If provided, the detections will be
            created from the predictions, and the format will be
            [class_id, x_center, y_center, width, height, confidence].
            Otherwise, from the labels.

        return
        ------
        a list of sv.Detections
        """
      
        detections = []
        # get file paths
        
        labels = [f for f in os.listdir(gt_label_path) if f.endswith(".txt")]
        labels = sorted([os.path.join(gt_label_path, f) for f in labels])
        
        images = self.get_images(gt_img_path)
        n_samples = len(images)
        # iterate each pair of image and label
        for i in range(n_samples):
            # get image info
            image = PIL.Image.open(images[i])
            img_w, img_h = image.size
            # get annotation
            label = labels[i]
            with open(label, "r") as f:
                lines = f.readlines()
                lines = [i.strip() for i in lines]
                
            if not lines:  # Checks if lines list is empty
                continue
            # each detection in the image/label
            ls_xyxy = []
            ls_cls = []
            ls_conf = []
            for i in lines:
                parts = i.split(" ")
                class_id = int(parts[0])
                coords = tuple(
                    map(float, parts[1:5])
                )  # x_center, y_center, width, height
                #conf = float(parts[5]) if path_preds else None
                
                xyxy = xywh2xyxy(
                    coords,
                    img_size=(img_w, img_h),
                )
                # append to lists
                ls_xyxy.append(xyxy)
                ls_cls.append(class_id)
                
            # create sv.Detections
            ls_xyxy = torch.stack(ls_xyxy).numpy()
            ls_cls = np.array(ls_cls)
            ls_conf = np.array(ls_conf)
            detection = sv.Detections(
                ls_xyxy,
                class_id=ls_cls,
                confidence= None,
            )
            detections.append(detection)
        return detections

    def get_pred_detections(self, path_preds):
        """
        get sv.Detections from labels

        params
        ------
        split: str
            "train" or "test"
        path_results: str
            path to the dir of predictions (.txt).
            If provided, the detections will be
            created from the predictions, and the format will be
            [class_id, x_center, y_center, width, height, confidence].
            Otherwise, from the labels.

        return
        ------
        a list of sv.Detections
        """
        #print(split)
        detections = []
        # get file paths
        
        labels = [f for f in os.listdir(path_preds) if f.endswith(".txt")]
        labels = sorted([os.path.join(path_preds, f) for f in labels])
      
        n_samples = len(labels)
        # iterate each pair of image and label
        for i in range(n_samples):
            # get image info
            # get annotation
            label = labels[i]
            with open(label, "r") as f:
                lines = f.readlines()
                lines = [i.strip() for i in lines]
                
            if not lines:
                continue
            
            # each detection in the image/label
            ls_xyxy = []
            ls_cls = []
            ls_conf = []
            for i in lines:
                parts = i.split(" ")
                #print(parts)
                class_id = int(parts[0])
                coords = tuple(
                    map(float, parts[1:5])
                )  # x_center, y_center, width, height
                conf = float(parts[5]) 
                #xyxy = xywh2xyxy(
                    #coords,
                    #img_size=(img_w, img_h),
                #)
                xyxy = torch.tensor(coords)
                # append to lists
                ls_xyxy.append(xyxy)
                ls_cls.append(class_id)
                ls_conf.append(conf)
            # create sv.Detections
            ls_xyxy = torch.stack(ls_xyxy).numpy()
            ls_cls = np.array(ls_cls)
            ls_conf_array = np.array(ls_conf)
            


            #print(ls_conf)
            #print(ls_conf_array)
            detection = sv.Detections(
                ls_xyxy,
                class_id=ls_cls,
                confidence=ls_conf_array,
            )
            detections.append(detection)
        return detections

    

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
