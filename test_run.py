import os
import time
import subprocess

# Function to create directory if it doesn't exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Base directory for all outputs
base_output_dir = "output_test_run"
ensure_dir(base_output_dir)

# Environment configuration - Adjust according to your script's needs
env_config = {
    "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.6,max_split_size_mb:128"
}

# Loop configurations similar to the Bash script
iterations = range(1, 3)  # Python ranges are exclusive on the end, so use 3 to include 2
n_train_values = [5, 20]
yolo_bases = ["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]

# Main loop to execute trial_nas.py with varying parameters
for i in iterations:
    for n_train in n_train_values:
        for yolo_base in yolo_bases:
            #suffix = 'ttt'
            #folder_name = f"{suffix}_{yolo_base}_{n_train}_{i}"
            
            suffix = f"aaa_{yolo_base}_{n_train}_{i}"
            #print(f"Iteration {i}, n_train {n_train}, model {yolo_base}, suffix {suffix}, folder_name {folder_name}")
            print(f"Iteration {i}, n_train {n_train}, model {yolo_base}, suffix {suffix}")
            ttt ='aaa'
            # Construct the command
            command = [
                "python", "trial_nas.py",
                "--iter", str(i),
                "--n_train", str(n_train),
                "--yolo_base", yolo_base,
                "--suffix", ttt
            ]
            
            # Run the command
            subprocess.run(command, env={**os.environ, **env_config})
            
            # Optional: sleep if needed between runs
            time.sleep(1)

# Note: Ensure trial_nas.py and other scripts are adapted to save outputs
# to the correct directories based on the 'suffix' or another method.
