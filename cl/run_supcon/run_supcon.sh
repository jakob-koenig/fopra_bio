#!/bin/bash -x
#SBATCH --account=hai_fzj_bda
#SBATCH --output=outs/%j.txt
#SBATCH --error=errs/%j.txt
#SBATCH --partition=booster
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1

module purge
module load Stages/2023
module load GCC
module load Python

source /p/project1/hai_fzj_bda/koenig8/jupyter/kernels/contrastive_learn/bin/activate
PYTHONPATH=/p/project1/hai_fzj_bda/koenig8/jupyter/kernels/contrastive_learn/lib/python3.10/site-packages/:$PYTHONPATH

srun python run_supcon.py
