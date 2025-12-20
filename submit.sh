#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A berzelius-2025-231
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J CL

# Load your environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Execute your code
python run.py --no-tune --model_name_extra brainmask --targets CL

#--model_name_extra brainmask --stratifycvby visual_read,gender

# --targets CL --input_cl CL
