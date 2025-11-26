#!/usr/bin/env python
"""Script to train all models sequentially."""

import os
import subprocess

# List of models to train
models = [
    "resnet50",
    "inception_v3",
    "vit_base_16",
    "bilinear_cnn",
    "mae_vit",
    "dino_vit",
    "ibot_vit"
]

def train_model(model_name):
    """Train a single model.
    
    :param model_name: Name of the model to train.
    """
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")
    
    # Command to train the model
    command = f"python src/train.py model={model_name} trainer=gpu data=my_dataset logger=tensorboard"
    
    # Execute the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Print the output
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    print(f"\n{'='*50}")
    print(f"Finished training {model_name}. Exit code: {result.returncode}")
    print(f"{'='*50}")
    
    return result.returncode

def main():
    """Main function to train all models."""
    print("Starting training of all models...")
    
    # Train each model
    for model in models:
        exit_code = train_model(model)
        if exit_code != 0:
            print(f"Error training {model}. Exiting...")
            break
    
    print("\nTraining of all models completed!")

if __name__ == "__main__":
    main()
