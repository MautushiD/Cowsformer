## This folder contains .sh files 
#### trialnas.sh : -[.sh file] (https://github.com/MautushiD/Cowsformer/blob/main/sh_scripts/trialnas.sh)
This script is designed for batch processing using the SLURM workload manager on a high-performance computing (HPC) environment. It is specifically tailored for running a series of experiments with a Python script (trial_nas.py) under different configurations.
##### Description: 
The script is a Shell script (#!/bin/sh) meant to be submitted as a job in an HPC environment. It utilizes various SLURM directives for resource allocation and job scheduling:

-p dgx_normal_q: Specifies the partition (or queue) to submit the job.
--account=niche_squad: Sets the account for billing and access control.
--time=48:10:00: Allocates a maximum runtime of 48 hours and 10 minutes for the job.
--nodes=1: Requests one compute node.
--ntasks-per-node=4: Defines the number of tasks to run on each node.
--gres=gpu:1: Requests one GPU per node.
--mem=32G: Allocates 32GB of memory for the job.
The script also includes environment setup commands for a Python environment:

Loading necessary modules (module load).
Activating a specific Anaconda environment (source activate cf).
Setting environment variables for PyTorch CUDA allocator configuration.
The core of the script is a nested loop structure to run the trial_nas.py script with different configurations:

Loop over a range of iterations (i from 1 to 300).
Inside this, loop over a set of training sizes (n_train with values 10, 25, 50, 100, 200).
Finally, loop over different model bases (yolo_base with values "yolo_nas_s", "yolo_nas_m", "yolo_nas_l").
Each configuration is executed with a python command, passing relevant arguments derived from the loops.

###### Usage
To use this script:

Ensure you are in an HPC environment with SLURM workload manager.
Load the required modules and activate the Python environment as described in the script.
Modify the trial_nas.py script path and any parameters as needed.
Submit the script to the SLURM queue using sbatch [script_name].sh

#### trialNas_single_run.sh

##### Description 
Description
This is a Bash script (#!/bin/bash) intended for submission as a job in an HPC environment. It includes various SLURM directives for resource allocation:

-p dgx_normal_q: Specifies the partition for job submission.
--account=niche_squad: Sets the account for job management and access control.
--time=119:59:59: Allocates a maximum runtime of nearly 120 hours.
--nodes=1: Requests one compute node.
--ntasks-per-node=4: Sets the number of tasks per node.
--gres=gpu:1: Requests one GPU resource per node.
--mem=32G: Allocates 32GB of memory for the job.
--output and --error: Directs standard output and error messages to specified files.
The script configures the environment for a Python job:

Loads necessary modules and the Anaconda3 environment.
Activates a specific Anaconda environment (cf).
Sets environment variables for PyTorch CUDA allocator.
The core functionality of the script is a loop to run the trial_nas.py script 1000 times with a fixed set of parameters:

Iterates over a range (i from 1 to 1000).
Each iteration runs the trial_nas.py script with a predefined number of training samples (n_train), model base (yolo_nas_l), and a unique suffix for each run.
