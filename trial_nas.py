import os
import argparse
import yaml
import torch

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# local imports
from models.nas import Niche_YOLO_NAS
from data.splitter.yolo import YOLO_Splitter

# constants
ROOT = os.path.dirname(__file__)
DIR_OUT = os.path.join(ROOT, "out")
DIR_MODEL = os.path.join(ROOT, "models")
DIR_DATA = os.path.join(ROOT, "data")
# Adjusted to match your directory structure
#DIR_COW200 = os.path.join(DIR_DATA, "cow200", "yolov5" )
#DIR_COW200 = os.path.join(DIR_DATA, "2_light")
DIR_COW200 = os.path.join(DIR_DATA, "1a_angle_t2s", "tv" )

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
    splitter = YOLO_Splitter(DIR_COW200, classes=["cow"], suffix=suffix+f"_{yolo_base}_{n_train}_{i}")
    
    #splitter = YOLO_Splitter(DIR_COW200, classes=["cow"], suffix=suffix)
    #splitter.shuffle_train_val(n_included=n_train)
    splitter.shuffle_train_val(n_included=n_train,k=5)
    path_data = splitter.write_dataset()
    #print("----------------------------------------------------------------------------")
    #print('path_data', path_data)
    #print("----------------------------------------------------------------------------")

    # log
    print("----------------------------------------------------------------------------")
    print(f"n_train: {n_train}, yolo_base: {yolo_base}, i: {i}, {suffix}")
    print("----------------------------------------------------------------------------")

    
    # variable batch sizes
    
    if n_train<=10:
        BATCH = 2
    elif 10<n_train<=50:
        BATCH = 5
    elif 50<n_train<=100:
        BATCH = 10
    elif 100<n_train:
        BATCH =16
    
        
    
    
    # define paths
    #name_task = f"n{n_train}_{yolo_base[:-3]}_i{i}_{suffix}"
    name_task = f"n{n_train}_{yolo_base}_i{i}"
    DIR_OUT_split = os.path.join(DIR_COW200, f'{suffix}_{yolo_base}_{n_train}_{i}')
    
    print('DIR_current',DIR_OUT_split)
    print("----------------------------------------------------------------------------")
    
    # configure model
    yolo_nas = Niche_YOLO_NAS(
        path_model=yolo_base,
        dir_train=os.path.join(DIR_OUT_split, "train"),
        dir_val=os.path.join(DIR_OUT_split, "val"),
        dir_test=os.path.join(DIR_OUT_split, "test"),
        name_task=name_task
    )
    
    
    
     
    # paths for the train and validation text files
    # os.path.join(DIR_COW200, 'train.txt')
    
    #path_train_txt = os.path.join(os.path.split(path_data)[0], 'train.txt')
    #path_val_txt = os.path.join(os.path.split(path_data)[0], 'val.txt') #os.path.join(DIR_COW200, 'val.txt')
    #path_test_txt = os.path.join(os.path.split(path_data)[0], 'test.txt') #os.path.join(DIR_COW200, 'test.txt')
    
    path_train_txt = os.path.join(DIR_OUT_split, 'train.txt')
    path_val_txt = os.path.join(DIR_OUT_split, 'val.txt') #os.path.join(DIR_COW200, 'val.txt')
    path_test_txt = os.path.join(DIR_OUT_split, 'test.txt') #os.path.join(DIR_COW200, 'test.txt')

    print("----------------------------------------------------------------------------")
    print('path_train_txt', path_train_txt)
    print('path_val_txt', path_val_txt)
    print("----------------------------------------------------------------------------")
    
    # path for the yaml file
    #path_yaml = os.path.join(os.path.split(path_data)[0], 'data.yaml') #os.path.join(DIR_COW200, 'data.yaml')
    path_yaml = os.path.join(DIR_OUT_split, 'data.yaml') #os.path.join(DIR_COW200, 'data.yaml')
    print('path_yaml', path_yaml)

    # train
    yolo_nas.train(path_yaml, path_train_txt, path_val_txt, BATCH, EPOCHS)

    

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, help="iteration number")
    parser.add_argument("--n_train", type=int,
                        help="number of images in training set")
    parser.add_argument("--yolo_base", type=str,
                        help="e.g., yolo8n, yolo8m, yolo8x")
    parser.add_argument("--suffix", type=str, help="suffix for folder name")
    #parser.add_argument("--dataset", type=str, help="e.g., 1a_angle_t2s, 1b_angle_s2t, 2_light, 3_breed, 4_all")
    args = parser.parse_args()
    main(args)