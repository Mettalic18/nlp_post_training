#!/bin/bash

#SBATCH --job-name=qwen3-squad-train
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00

# Change 'rbunescu' to your <username>
cd /users/mmajeske/hw06/hw06/sft

# Make sure the right Python environment is activated before running this.
python sft_squad.py -train -model_load /projects/class/itcs6101_091/hw06/models/Qwen3-0.6B-Base