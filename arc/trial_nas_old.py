import os
import argparse
import yaml

# local imports
from models.nas import Niche_YOLO_NAS
# Adjusted the import to reflect your directory structure
from data.splitter.yolo import YOLO_Splitter

# constants
ROOT = os.path.dirname(__file__)
DIR_OUT = os.path.join(ROOT, "out")
DIR_MODEL = os.path.join(ROOT, "models")
DIR_DATA = os.path.join(ROOT, "data")
# Adjusted to match your directory structure
DIR_COW200 = os.path.join(DIR_DATA, "cow200", "yolov5")

# model configuration
BATCH = 16
EPOCHS = 100


def main(args):
    # parse arguments
    i = args.iter
    n_train = args.n_train
    yolo_base = args.yolo_base
    suffix = args.suffix

    # shuffle dataset
    splitter = YOLO_Splitter(DIR_COW200, classes=["cow"], suffix=suffix)
    splitter.shuffle_train_val(n_included=n_train)
    path_data = splitter.write_dataset()
    #print('path_data', path_data)

    # log
    print("-----------------------------------")
    print(f"n_train: {n_train}, yolo_base: {yolo_base}, i: {i}, {suffix}")
    print("-----------------------------------")

    # define paths
    name_task = f"n{n_train}_{yolo_base[:-3]}_i{i}_{suffix}"

    # configure model
    yolo_nas = Niche_YOLO_NAS(
        path_model=yolo_base,
        dir_train=os.path.join(DIR_OUT, "train"),
        dir_val=os.path.join(DIR_OUT, "val"),
        name_task=name_task
    )

    # paths for the train and validation text files
    # os.path.join(DIR_COW200, 'train.txt')
    path_train_txt = os.path.join(os.path.split(path_data)[0], 'train.txt')
    path_val_txt = os.path.join(os.path.split(path_data)[0], 'val.txt') #os.path.join(DIR_COW200, 'val.txt')

    # path for the yaml file
    path_yaml = os.path.join(os.path.split(path_data)[0], 'data.yaml') #os.path.join(DIR_COW200, 'data.yaml')
    print('path_yaml', path_yaml)

    # train
    yolo_nas.train(path_yaml, path_train_txt, path_val_txt, BATCH, EPOCHS)

    # evaluate
    yolo_nas.evaluate()


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, help="iteration number")
    parser.add_argument("--n_train", type=int,
                        help="number of images in training set")
    parser.add_argument("--yolo_base", type=str,
                        help="e.g., yolo8n, yolo8m, yolo8x")
    parser.add_argument("--suffix", type=str, help="suffix for folder name")
    args = parser.parse_args()
    main(args)
#