#!/bin/bash -x
#SBATCH --account=hai_fzj_bda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --output=outs/%j.txt
#SBATCH --error=errs/%j.txt
#SBATCH --partition=booster
#SBATCH --time=04:30:00
#SBATCH --gres=gpu:1

# *** start of job script **
module load Stages/2023
module load GCCcore/.11.3.0
module load Python/3.10.4

source $PROJECT/koenig8/jupyter/kernels/opt_transport/bin/activate
PYTHONPATH=/p/project1/hai_fzj_bda/koenig8/jupyter/kernels/opt_transport/lib/python3.10/site-packages/:$PYTHONPATH

srun python -u translate.py -s "$PROJECT/koenig8/ot/data/adata_st.h5ad" -t "$PROJECT/koenig8/ot/data/adata_histo.h5ad" -p "$PROJECT/koenig8/ot/models/normalized_fused" --source_key "pca_plus_slides_scaled" --target_key "uni_pca_plus_coords" --linear_term brain_area_onehot --sample_target

cp outs/$SLURM_JOB_ID.txt "$PROJECT/koenig8/ot/models/sampled_targets/log.txt"
