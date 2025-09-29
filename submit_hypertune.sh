#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A berzelius-2025-231
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH -J CNN-VR-tune


# Load your environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Execute your code
python run.py --no-tune --targets CL --model_name_extra CL_2split
srun --ntasks=8 --gres=gpu:1 --cpus-per-task=8 \
  bash -lc 'CUDA_VISIBLE_DEVICES=$SLURM_LOCALID \
    python run.py \
      --targets CL\
      --model_name_extra CL_tune \
      --storage sqlite:////scratc/proj/berzelius-2024-156/users/x_yxiao/AI-PET/scripts/optuna_ai_pet.db \
      --n_trials 999999 \
      --tune_timeout 86400 \
      --proxy_epochs 6 \
      --proxy_folds 3 \
      --batch_size 2 \
      --amp \
      --num_workers 8'
