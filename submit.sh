#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A berzelius-2025-231
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH -J VR-mask

# Load your environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Execute your code
python run.py --no-tune --model_name_extra brainmask

#--model_name_extra brainmask --stratifycvby visual_read,gender

# --Targets CL
