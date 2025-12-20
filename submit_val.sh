#!/bin/bash
# SLURM batch job script for Berzelius

#SBATCH -A berzelius-2025-231
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH -J CL-val

# Load your environment
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate ai-pet

# Execute your code
python run_val.py --targets CL --dataset IDEAS --best_model_folder CNN3D_CL_2split80-20_stratify-visual_read,site_brainmask_20251216_232720

# python run_val.py --dataset IDEAS --best_model_folder CNN3D_visual_read_2split80-20_stratify-visual_read,site_brainmask_20251216_232629
# CNN3D_visual_read_2split80-20_stratify-visual_read,site_IDEAS_Inten_Norm_20251004_022211
# CNN3D_visual_read_2split80-20_stratify-visual_read,site_brainmask_extra-lastlayer-input-CL_20251216_232707
# --targets CL
