#!/bin/bash
#SBATCH -p dgx_normal_q
#SBATCH --account=niche_squad
#SBATCH --time=119:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=_single_run.out
#SBATCH --error=_single_run.err


# Set PyTorch CUDA allocator configuration if needed
module load site/tinkercliffs/easybuild/setup
module load Anaconda3/2020.11
source activate cf
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

# Nested loops to run the Python script with different configurations
for i in {1..1000}; 
do
    echo "Iteration $i"
    suffix="exp_${yolo_base}_${n_train}_${i}"
    python trial_nas.py  \
        --iter $i \
        --n_train 100\
        --yolo_base "yolo_nas_l" \
        --suffix $suffix
done
