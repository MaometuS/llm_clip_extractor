#!/bin/bash
#SBATCH --job-name=DION_IAD
###########RESOURCES###########
#SBATCH --partition=48-4
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
###############################
#SBATCH --output=TEST.out
#SBATCH --error=TEST.err
#SBATCH -v

source ~/anaconda3/etc/profile.d/conda.sh
conda activate clip_extractor
srun python extractor.py
conda deactivate