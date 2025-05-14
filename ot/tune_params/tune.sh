#!/bin/bash -x
#SBATCH --account=hai_fzj_bda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=outs/%j.txt
#SBATCH --error=errs/%j.txt
#SBATCH --partition=booster
#SBATCH --time=9:30:00
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1

# *** start of job script **
module load Stages/2023
module load GCCcore/.11.3.0
module load Python/3.10.4

source $PROJECT/koenig8/jupyter/kernels/opt_transport/bin/activate
PYTHONPATH=/p/project1/hai_fzj_bda/koenig8/jupyter/kernels/opt_transport/lib/python3.10/site-packages/:$PYTHONPATH

srun python tune.py -s "$PROJECT/koenig8/ot/data/adata_st.h5ad" -t "$PROJECT/koenig8/ot/data/adata_histo.h5ad" -o "$PROJECT/koenig8/ot/tune_params/only_scaled" --source_key pca_plus_slides_scaled --target_key "uni_pca_plus_coords" --linear_term brain_area_similarities --metric IDWE --cost cosine
