#!/bin/bash

#SBATCH --job-name=qwen25-grpo-gsm
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00

cd /users/rbunescu/6101/hw06/grpo
python grpo_gsm.py -train -model_load /projects/class/itcs6101_091/hw06/models/Qwen2.5-0.5B-Instruct
