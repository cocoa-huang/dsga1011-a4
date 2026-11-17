#!/bin/bash
#SBATCH --account=ds_ga_1011-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=%j_%x.out
#SBATCH --requeue
# Usage (from part1/): sbatch run.sbatch --train --eval

MF=/scratch/$USER/miniforge3
source $MF/etc/profile.d/conda.sh
conda activate a4p1
export HF_HOME=/scratch/$USER/hf_cache
cd /scratch/$USER/a4/part1
python -c 'import torch; print("GPU:", torch.cuda.get_device_name())'
python3 main.py "$@"
