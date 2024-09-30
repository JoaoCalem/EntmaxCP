#!/bin/bash
#SBATCH --job-name=softmax    # Job name
#SBATCH --output="job.%x.%j.out" # Name of stdout output file (%x expands to job name and %j expands to %jobId)
#SBATCH --gres=gpu:1            # Number of GPUs to be used
#SBATCH --qos=gpu-medium         # QOS to be used
#SBATCH --error="job.%x.%j.err"
set -eu
set -x

cd /home/margarida/EntmaxCP
source /home/margarida/.profile
pyenv activate entmaxcp

# Define the list of models, datasets, and losses to loop through
#models=("cnn" "vit")  # Add models if needed
dataset="ImageNet"  # Replace with your datasets
loss="softmax"  # Replace with your loss functions
seeds=(05 19 95 42) # Replace with your seeds
# Define the other optional parameters
epochs=10  # Default number of epochs
patience=2  # Default patience
model="vit"  # Change to "vit"
for seed in "${seeds[@]}"
do
# Create a unique save filename based on model, dataset, and loss
    save_filename="/mnt/data-poseidon/margarida/training/${model}_${dataset}_${loss}_${seed}_${epochs}_model"

    # Run the Python script with the current arguments
    python example_usage/train.py "$model" "$dataset" "$loss" "$save_filename" --seed "$seed" --epochs "$epochs" --patience "$patience"

    # Optionally print a message after each run (for debugging/logging purposes)
    echo "Run with model=$model, dataset=$dataset, loss=$loss, seed=$seed completed."
done