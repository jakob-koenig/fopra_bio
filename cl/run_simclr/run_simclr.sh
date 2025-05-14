#!/bin/bash -x
#SBATCH --account=hai_fzj_bda
#SBATCH --output=outs/%j.txt
#SBATCH --error=errs/%j.txt
#SBATCH --partition=booster
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

module purge
module load Stages/2023
module load GCC
module load Python

source /p/project1/hai_fzj_bda/koenig8/jupyter/kernels/contrastive_learn/bin/activate
PYTHONPATH=/p/project1/hai_fzj_bda/koenig8/jupyter/kernels/contrastive_learn/lib/python3.10/site-packages/:$PYTHONPATH

srun python run_simclr.py -w sup_con_one -s "$PROJECT/koenig8/ot/data/adata_st.h5ad" -t "$PROJECT/koenig8/ot/data/adata_histo.h5ad" --target_key uni_pca_95 --lr 2e-4 --dropout 0.2 --mlp_layers "700,700,64" --max_epochs 20 -p 256 -n 1024 --weight_decay 0.001 --projection_head