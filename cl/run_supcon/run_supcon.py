import os
import sys

current_dir = os.path.dirname(__file__)  # cl/run_sim_clr
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from simple_model.model import SimCLR, train_simclr
from simple_model.dataloader import PairedContrastiveDataset, make_sampler
import pytorch_lightning as pl
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import default_restore_location
import matplotlib.pyplot as plt
from numpy._core.multiarray import scalar

import scanpy as sc

path = "/p/project1/hai_fzj_bda/koenig8/ot/data/"
adata_st = sc.read_h5ad(os.path.join(path, "adata_st.h5ad"))
adata_histo = sc.read_h5ad(os.path.join(path, "adata_histo.h5ad"))

train_set = make_sampler(
    adata_st, adata_histo, 10, 150, oversample_fraction=0.3, st_key = "pca_embedding", histo_key = "uni_pca_95", 
    class_key = "brain_area_onehot", mode = "train", seed = 42
)
val_set = make_sampler(
    adata_st, adata_histo, 5, 100, oversample_fraction=0.3, st_key = "pca_embedding", histo_key = "uni_pca_95", 
    class_key = "brain_area_onehot", mode = "val", seed = 42
)

torch.serialization.add_safe_globals([
    torch._utils._rebuild_tensor_v2,  
    np._core.multiarray.scalar
])

search_space = [
    Integer(64, 1024, name='hd1'),
    Integer(64, 1024, name='hd2'),
    Integer(16, 256, name='ld'),               
    Real(1e-5, 1e-2, prior='log-uniform', name='lr'),
    Real(1e-6, 1e-3, prior='log-uniform', name='wd'),
    Categorical([True, False], name='dropout'),
    Categorical([True, False], name='ph'),
    Categorical([True, False], name='batchnorm')
]

# Objective function
@use_named_args(search_space)
def objective(**params):
    workdir = "/p/project1/hai_fzj_bda/koenig8/cl/supcon_2"
    logger = pl.loggers.TensorBoardLogger(
        os.path.join(workdir, "logs"), name="SupCon"
    )

    param_str = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
    print(f"New trial with params: \n{param_str}")

    
    model, val_acc = train_simclr(
        train_set, val_set,
        CHECKPOINT_PATH=os.path.join(workdir, "simclr_ckpt"),
        logger=logger,
        monitor="val_acc_mean_pos", monitor_mode="max",
        return_monitored = True,
        max_epochs=130, patience=10,
        hidden_dims=[int(params['hd1']), int(params['hd2'])],
        latent_dim=int(params['ld']),
        lr=float(params['lr']),
        weight_decay=float(params['wd']),
        dropout=bool(params['dropout']),
        projection_head=bool(params['ph']),
        use_batchnorm=bool(params['batchnorm']),
        temperature=0.07,
        log_every=6000, refresh_rate=0,
        shuffle_every=1,     
        seed=42
    )
    print("Val acc:", -val_acc)
    return -val_acc

# Launch optimization
result = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=100,
    random_state=42,
    n_initial_points=5
)

print("Best score: ", -result.fun)
print("Best hyperparameters:")
for name, val in zip([dim.name for dim in search_space], result.x):
    print(f"{name}: {val}")