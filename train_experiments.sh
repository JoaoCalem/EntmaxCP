#!/bin/bash

# Define the list of models, datasets, and losses to loop through
#models=("MNIST" "CIFAR10" "CIFAR100")  # Replace with your models
datasets=("CIFAR10","CIFAR100")  # Replace with your datasets
losses=("entmax" "sparsemax" "softmax")  # Replace with your loss functions
seeds=(23 05 19 95 42)  # Replace with your seeds
# Define the other optional parameters
epochs=20  # Default number of epochs
patience=3  # Default patience
model="vit"
# Loop through each combination of model, dataset, and loss
for loss in "${losses[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for seed in "${seeds[@]}"
        do
            # Create a unique save filename based on model, dataset, and loss
            save_filename="${model}_${dataset}_${loss}_${seed}_${epochs}_model"

            # Run the Python script with the current arguments
            python example_usage/train.py "$model" "$dataset" "$loss" "$save_filename" --seed "$seed" --epochs "$epochs" --patience "$patience"

            # Optionally print a message after each run (for debugging/logging purposes)
            echo "Run with model=$model, dataset=$dataset, loss=$loss completed."
        done
    done
done
