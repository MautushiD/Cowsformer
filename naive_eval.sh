#!/bin/bash
#SBATCH -p a100_normal_q
#SBATCH --account=vos
#SBATCH --time=48:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=out.log
#SBATCH --error=out.err

module load site/tinkercliffs-rome_a100/easybuild/setup
module load Anaconda3/2020.11
source activate tf
 
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

python3.9 /home/niche/cowsformer/naive_eval.py
