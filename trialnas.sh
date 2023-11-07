#!/bin/bash

# Set PyTorch CUDA allocator configuration if needed
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

# Nested loops to run the Python script with different configurations
for i in {1..40}; do
    for n_train in 10 25; do
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

