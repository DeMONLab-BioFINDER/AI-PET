#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A berzelius-2025-231
#SBATCH --gpus=1
#SBATCH -t 06:00:00
#SBATCH -J viz-23015

# Load your environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Execute your code
python run_vis.py --best_model_folder IDEAS_ADNI_CNN3D_visual_read_2split80-20_stratify-dataset,visual_read_Balanced_sampling_L1-10_all-scans_20260106_034136 --visualization_name occlusion --vis_img_list 23015

#21628,20448,21156,21764,21884,22530,22696,23015