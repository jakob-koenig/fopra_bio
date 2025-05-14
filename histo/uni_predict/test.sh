#!/bin/bash -x
#SBATCH --account=hai_fzj_bda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=htest.txt
#SBATCH --error=test.txt
#SBATCH --partition=booster
#SBATCH --time=04:30:00
#SBATCH --hint=nomultithread
#SBATCH --gres=none  # Ensure no GPUs are allocated

# *** start of job script **

sinfo -O gres
