#!/bin/bash -x
#SBATCH --account=hai_fzj_bda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=outs/%j.txt
#SBATCH --error=errs/%j.txt
#SBATCH --partition=booster
#SBATCH --time=04:30:00
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1

# *** start of job script **
module load Stages/2023
module load GCCcore/.11.3.0
module load Python/3.10.4

source $PROJECT/koenig8/jupyter/kernels/uni/bin/activate
PYTHONPATH=/p/project1/hai_fzj_bda/koenig8/jupyter/kernels/uni/lib/python3.10/site-packages/:$PYTHONPATH

srun python uni_predict.py "$PROJECT/koenig8/histo/embeddings_scaled_768" "$PROJECT/koenig8/histo/data/scaled_images" 768
