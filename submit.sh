#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A berzelius-2025-231
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH -J VR

# Load your environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Execute your code
python run.py --no-tune --model_name_extra Balanced_sampling_L1-10_all-scans --stratifycvby dataset,visual_read --samesubject_col sameID

#--model_name_extra brainmask --stratifycvby visual_read,gender --model UNet3D

# --targets CL --input_cl CL --smoothl1_beta 5 --reg_loss mse --extra_global_feats p95,std,frac_hi
