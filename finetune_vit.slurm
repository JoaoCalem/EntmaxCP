#!/bin/bash
#SBATCH --job-name=vit_tune    # Job name
#SBATCH --output="job.%x.%j.out" # Name of stdout output file (%x expands to job name and %j expands to %jobId)
#SBATCH --gres=gpu:1            # Number of GPUs to be used
#SBATCH --qos=gpu-medium         # QOS to be used
#SBATCH --error="job.%x.%j.err"
set -eu
set -x

cd /home/margarida/EntmaxCP
source /home/margarida/.profile
source /home/margarida/.bashrc

echo $PATH
hostname
pyenv activate entmaxcp
# Define the list of models, datasets, and losses to loop through
#models=("cnn" "vit")  # Add models if needed
datasets=("NewsGroups")  # Replace with your datasets
losses=("entmax" "sparsemax" "softmax")  # Replace with your loss functions
seeds=(23 05 19 95 42)
# Define the other optional parameters
epochs=20  # Default number of epochs
patience=2  # Default patience
model="bert"  # Change to "vit"


for loss in "${losses[@]}"
do
    for dataset in "${datasets[@]}"
    do
            # Loop through all seeds for cnn model
        for seed in "${seeds[@]}"
        do
            # Create a unique save filename based on model, dataset, and loss
            save_filename="/mnt/data-poseidon/margarida/training/${model}_${dataset}_${loss}_${seed}_${epochs}_model"

            # Run the Python script with the current arguments
            python example_usage/train.py "$model" "$dataset" "$loss" "$save_filename" --seed "$seed" --epochs "$epochs" --patience "$patience"

            # Optionally print a message after each run (for debugging/logging purposes)
            echo "Run with model=$model, dataset=$dataset, loss=$loss, seed=$seed completed."
        done
    done
done