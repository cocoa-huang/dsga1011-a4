#!/bin/bash
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=%j_%x.out
#SBATCH --requeue
# Usage (from part2/): sbatch run.sbatch --finetune --learning_rate 3e-4 ... --use_wandb --experiment_name NAME

source /scratch/$USER/miniforge3/etc/profile.d/conda.sh
conda activate a4p2
export HF_HOME=/scratch/$USER/hf_cache WANDB_DIR=/scratch/$USER TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$(cat /scratch/$USER/.wandb_key)
cd /scratch/$USER/a4/part2
python -c 'import torch; print("GPU:", torch.cuda.get_device_name())'
python train_t5.py "$@"
