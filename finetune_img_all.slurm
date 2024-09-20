#!/bin/bash
#SBATCH --job-name=imgnet_all    # Job name
#SBATCH --output="job.%x.%j.out" # Name of stdout output file (%x expands to job name and %j expands to %jobId)
#SBATCH --gres=gpu:1            # Number of GPUs to be used
#SBATCH --qos=gpu-medium         # QOS to be used
#SBATCH --error="job.%x.%j.err"
set -eu
set -x

cd /home/margarida/EntmaxCP
source /home/margarida/.profile
source /home/margarida/.bashrc

pyenv activate entmaxcp
# Define the list of models, datasets, and losses to loop through
#models=("cnn" "vit")  # Add models if needed
dataset="ImageNet"  # Replace with your datasets
losses=("entmax" "softmax" "sparsemax") # Replace with your loss functions
seed=23 # Replace with your seeds
# Define the other optional parameters
epochs=5  # Default number of epochs
patience=2  # Default patience
model="vit"  # Change to "vit"

for loss in "${losses[@]}"; do
     
    # Create a unique save filename based on model, dataset, and loss
    save_filename="/mnt/data-poseidon/margarida/training/${model}_${dataset}_${loss}_${seed}_${epochs}_model/"
    
    mkdir -p $save_filename

    # Run the Python script with the current arguments
    srun --output="$save_filename/$loss.out" --error="$save_filename/$loss.err" python example_usage/train.py "$model" "$dataset" "$loss" "$save_filename" --seed "$seed" --epochs "$epochs" --patience "$patience" &

    # Optionally print a message after each run (for debugging/logging purposes)
    echo "Run with model=$model, dataset=$dataset, loss=$loss, seed=$seed completed."
done

wait