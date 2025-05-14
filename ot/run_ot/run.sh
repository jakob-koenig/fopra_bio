#!/bin/bash -x
#SBATCH --account=hai_fzj_bda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=outs/%j.txt
#SBATCH --error=errs/%j.txt
#SBATCH --partition=booster
#SBATCH --time=06:30:00
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1

# *** start of job script **
module load Stages/2023
module load GCCcore/.11.3.0
module load Python/3.10.4

source $PROJECT/koenig8/jupyter/kernels/opt_transport/bin/activate
PYTHONPATH=/p/project1/hai_fzj_bda/koenig8/jupyter/kernels/opt_transport/lib/python3.10/site-packages/:$PYTHONPATH

srun -u python run.py -s "$PROJECT/koenig8/ot/data/adata_st.h5ad" -t "$PROJECT/koenig8/ot/data/adata_histo.h5ad" -p "$PROJECT/koenig8/ot/models/unfused" --source_key "pca_embedding" --target_key "uni_pca_95" --alpha "1.0" --epsilon "1e-3" --sample_target
