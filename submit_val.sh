#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A berzelius-2025-231
#SBATCH --gpus=1
#SBATCH -t 06:00:00
#SBATCH -J VR_5shot

# Load your environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Execute your code
python run_val.py --dataset A4  --few_shot 5 --best_model_folder IDEAS_ADNI_UNet3D_visual_read_2split80-20_stratify-dataset,visual_read_Balanced_sampling_L1-10_all-scans_20260106_034136 --model UNet3D --unfreeze_layers 1 --few_shot_iterations 10
# python run_val.py --dataset IDEAS --best_model_folder IDEAS_ADNI_CNN3D_CL_2split80-20_stratify-dataset,CL_Balanced_sampling_L1-10_all-scans_20260106_033903
# IDEAS_ADNI_CNN3D_visual_read_2split80-20_stratify-dataset,visual_read_Balanced_sampling_L1-10_all-scans_20260106_034136
# --targets CL --few_shot 5 --unfreeze_layers 1 --few_shot_iterations 1 --stratifycvby visual_read
# --model Unet3D --unfreeze_layers 3 --few_shot_iterations 10