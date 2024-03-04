#!/bin/sh
#SBATCH -p dgx_normal_q
#SBATCH --account=niche_squad
<<<<<<< Updated upstream
#SBATCH --time=48:10:00
=======
#SBATCH --time=90:10:00
>>>>>>> Stashed changes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G


# Set PyTorch CUDA allocator configuration if needed
<<<<<<< Updated upstream
module load site/tinkercliffs/easybuild/setup
module load Anaconda3/2020.11
source activate cf
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

# Nested loops to run the Python script with different configurations
for i in {1..300}; do
=======
#module load site/tinkercliffs/easybuild/setup
#module load Anaconda3/2020.11

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

# Nested loops to run the Python script with different configurations
for i in {1..10}; do
>>>>>>> Stashed changes
    for n_train in 10 25 50 100 200 ; do
        for yolo_base in "yolo_nas_s" "yolo_nas_m" "yolo_nas_l"; do
            suffix="exp_${yolo_base}_${n_train}_${i}"
            echo "Iteration $i, n_train $n_train, model $yolo_base, suffix $suffix"
            python trial_nas.py \
                --iter $i \
                --n_train $n_train \
                --yolo_base $yolo_base \
                --suffix $suffix
            sleep 1
        done
    done
done

