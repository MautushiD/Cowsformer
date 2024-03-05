import os
os.chdir("..")
ROOT = os.getcwd()
#os.environ['TORCH_HOME'] = '/home/mautushid/.torch'
from ultralytics import NAS
from models.nas import *
import argparse
import supervision as sv
from API import*
from evaluate import from_sv
import pandas as pd




def main(args):
    # parse arguments
    yolo_base = args.yolo_base
    config = args.config
    exp_name = args.exp_name
    n = args.n
    iteration = args.iteration
    config_short = config.split("_")[-1]

    dir_train = ROOT+ "/data/"+config+"/tv/"+ exp_name+"_"+ yolo_base + "_" + \
            str(n) + "_" + str(iteration) + "_" + config_short+ "_" + yolo_base +"_" + str(n) + "_" + str(iteration) + "/"+ "train"
    dir_val = ROOT+ "/data/"+config+"/tv/"+ exp_name+"_"+ yolo_base + "_" + \
                str(n) + "_" + str(iteration) + "_" + config_short+ "_" + yolo_base +"_" + str(n) + "_" + str(iteration) + "/"+ "val"
    dir_test = ROOT+ "/data/"+config + "/test"

    data_yaml_path = ROOT+ "/data/"+config+"/tv/"+ exp_name+"_"+ yolo_base + "_" + \
                str(n) + "_" + str(iteration) + "_" + config_short+ "_" + yolo_base +"_" + str(n) + "_" + str(iteration) + "/"+ "data.yaml"
    base_dir = ROOT + "/checkpoints/n" + str(n) + "_" + yolo_base + "_i" + str(iteration) + "_" + config_short
    items_under_base = os.listdir(base_dir)[0]
    finetuned_model_path = base_dir + "/"+ items_under_base + "/ckpt_best.pth"
    output_dir = dir_test +"/"+ yolo_base +"_"+ str(n)+"_"+str(iteration)+ "_labelsPred"

    
    ### Creating instance of Niche_YOLO_NAS class
    my_nas = Niche_YOLO_NAS(yolo_base, dir_train, dir_val, dir_test, "cow200")
    predictions = my_nas.prediction(data_yaml_path,finetuned_model_path)
    my_nas.write_predictions(predictions,output_dir)



if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_base", type=str, help="yolo_nas_l,yolo_nas_m")
    parser.add_argument("--config", type=str,
                        help="1a_angle_t2s, 1b_angle_s2t, 2_light, 3_breed, 4_all")
    parser.add_argument("--exp_name", type=str,
                        help="exp")
    parser.add_argument("--n", type=int, help="16, 32,64, 128...")
    parser.add_argument("--iteration", type=int, help="1,2,3...")
    args = parser.parse_args()
    main(args)