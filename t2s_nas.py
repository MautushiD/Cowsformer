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
DIR_COW200 = os.path.join(DIR_DATA, "1a_angle_t2s", "tv" ) ##need to change for each dataset --------------------

# model configuration
BATCH = 16
EPOCHS = 100


def main(args):
    # parse arguments
    i = args.iter
    n_train = args.n_train
    yolo_base = args.yolo_base
    suffix = args.suffix
    
    ### run the following when .sh
    folder_name_string = suffix+"_t2s"+f"_{yolo_base}_{n_train}_{i}"  ##need to change for each dataset --------------------
    
    ### run the following when in test_run.py
    #folder_name_string = suffix+f"_{yolo_base}_{n_train}_{i}" + '_t2s_'+yolo_base+ '_' + str(n_train) + '_' + str(i)##need to change for each dataset --------------------
    
    # shuffle dataset
    splitter = YOLO_Splitter(DIR_COW200, classes=["cow"], suffix=folder_name_string)
    
    #splitter = YOLO_Splitter(DIR_COW200, classes=["cow"], suffix=suffix)
    #splitter.shuffle_train_val(n_included=n_train)
    splitter.shuffle_train_val(n_included=n_train,k=5)
    path_data = splitter.write_dataset()
    
    # log
    print("----------------------------------------------------------------------------")
    print(f"n_train: {n_train}, t2s_yolo_base: {yolo_base}, i: {i}, {suffix}")
    print("----------------------------------------------------------------------------")

    if n_train <= 16:
        BATCH = 8
    else:
        BATCH = 16
    
    # define paths
    #name_task = f"n{n_train}_{yolo_base[:-3]}_i{i}_{suffix}"
    name_task = f"n{n_train}_{yolo_base}_i{i}"+"_t2s"   #### need to change for each dataset ------------------------
    
    DIR_OUT_split = os.path.join(DIR_COW200, folder_name_string)
    
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
    
    checkpoint_dir = ROOT + '/checkpoints/' + 'n' + \
        str(n_train) + '_' + yolo_base + '_' + 'i' + str(i) + '_t2s'
    path_train_txt = os.path.join(DIR_OUT_split, 'train.txt')
    path_val_txt = os.path.join(DIR_OUT_split, 'val.txt') #os.path.join(DIR_COW200, 'val.txt')
    path_test_txt = os.path.join(DIR_OUT_split, 'test.txt') #os.path.join(DIR_COW200, 'test.txt')

    print("----------------------------------------------------------------------------")
    print('path_train_txt', path_train_txt)
    print('path_val_txt', path_val_txt)
    print("----------------------------------------------------------------------------")
    
    # path for the yaml file
    #path_yaml = os.path.join(os.path.split(path_data)[0], 'data.yaml') #os.path.join(DIR_COW200, 'data.yaml')
    path_yaml = os.path.join(DIR_OUT_split, 'data.yaml')
    print('path_yaml', path_yaml)

    # train
    yolo_nas.train(path_yaml, path_train_txt, path_val_txt, BATCH, EPOCHS)
    
    # new function to keep only the best_ckpt.pth
    yolo_nas.remove_ckpt(checkpoint_dir,'latest')

    # perfom evaluation
    yolo_nas.evaluate_test_set(ROOT, yolo_base, '1a_angle_t2s', 'exp', n_train, i)
    
    ## remove chekpoint best
    yolo_nas.remove_ckpt(checkpoint_dir, 'best')

    

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