#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A berzelius-2025-231
#SBATCH --gpus=2
#SBATCH --ntasks=1                 # one launcher task
#SBATCH -t 3-00:00:00
#SBATCH -J CL-tune


# Load your environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Execute your code
srun --ntasks=2 --gres=gpu:1 --cpus-per-task=8 \
  bash -lc 'CUDA_VISIBLE_DEVICES=$SLURM_LOCALID \
    python run.py --targets CL \
      --ddp 1 \
      --storage sqlite:////scratc/proj/berzelius-2024-156/users/x_yxiao/AI-PET/scripts/optuna_ai_pet_CL.db \
      --n_trials 999999 \
      --tune_timeout 86400 \
      --num_workers 2 '

# --targets CL
#SBATCH -N 1
#SBATCH --gpus-per-node=8