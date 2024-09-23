#!/bin/bash

# Define the list of models, datasets, and losses to loop through
#models=("cnn" "vit")  # Add models if needed
datasets=("NewsGroups")  # Replace with your datasets
losses=("entmax" "sparsemax" "softmax")  # Replace with your loss functions
seeds=(23 05 19 95 42)  # Replace with your seeds
# Define the other optional parameters
epochs=20  # Default number of epochs
patience=2  # Default patience
model="vit"  # Change to "vit" 

# Loop through each combination of model, dataset, and loss
for loss in "${losses[@]}"
do
    for dataset in "${datasets[@]}"
    do
            # Loop through all seeds for cnn model
        for seed in "${seeds[@]}"
        do
            # Create a unique save filename based on model, dataset, and loss
            save_filename="/mnt/data-poseidon/margarida/training/${model}_${dataset}_${loss}_${seed}_${epochs}"

            # Run the Python script with the current arguments
            python example_usage/train.py "$model" "$dataset" "$loss" "$save_filename" --seed "$seed" --epochs "$epochs" --patience "$patience"

            # Optionally print a message after each run (for debugging/logging purposes)
            echo "Run with model=$model, dataset=$dataset, loss=$loss, seed=$seed completed."
        done
    done
done