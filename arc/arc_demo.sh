#!/bin/bash
#SBATCH -p dgx_normal_q
#SBATCH --account=niche_squad
#SBATCH --time=48:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

python3.9 trial_nas.py \
        --iter $i\
        --n_train $n_train\
        --yolo_base $yolo_base\
        --suffix $suffix
# when you want to run this script, type:
# sbatch arc_demo.sh

# trial_nas.py: