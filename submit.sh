#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A berzelius-2025-231
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH -J CL-G

# Load your environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Execute your code
python run.py --no-tune  --tragets CL --model_name_extra CL_2split-gender

#--model_name_extra visual_read_2split-VRG --stratifycvby visual_read,gender

# --tragets CL
