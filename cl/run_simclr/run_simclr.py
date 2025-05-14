import os
import sys

current_dir = os.path.dirname(__file__)  # cl/run_sim_clr
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)
from simple_model.model import train_simclr, train_coordinate_regressor
from simple_model.dataloader import PairedContrastiveDataset, CoordinateDataset

import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import scanpy as sc

import argparse

# Argument parsing ========================================
parser = argparse.ArgumentParser(description="Training configuration")

# Required: working directory path
parser.add_argument(
    "-w",
    "--workdir", 
    type=str, 
    required=True, 
    help="Working directory for saving outputs and logs"
)

parser.add_argument("-s", "--source", type=str, required=True, help="Path to the input source file")
parser.add_argument("-t", "--target", type=str, required=True, help="Path to the input target file")
parser.add_argument("-q", "--source_key", type=str, default = "pca_embedding", help="Key in source adata obsm")
parser.add_argument("-r", "--target_key", type=str, default = "uni_embedding", help="Key in target adata obsm")
parser.add_argument("-p", "--pos", type=int, default=30, help="# of Positive samples in SimCLR")
parser.add_argument("-n", "--neg", type=int, default=150, help="# of Negatives samples in SimCLR")
parser.add_argument("--monitor", type=str, default="val_loss", help="Monitor for saving checkpoints ")
parser.add_argument("--projection_head",action="store_true", help="Use L2 Norm before final layer of SimCLR")

# Optional: dropout rate (float)
parser.add_argument(
    "--dropout", 
    type=float, 
    default=0.0, 
    help="Dropout rate (default: 0.0)"
)

# Optional: max epochs (int)
parser.add_argument(
    "--max_epochs", 
    type=int, 
    default=500, 
    help="Max epochs SimCLR (default: 500)"
)

# Optional: use batch normalization (bool)
parser.add_argument(
    "--batchnorm", 
    action="store_true", 
    help="Enable batch normalization"
)

# Optional: list of MLP hidden layer sizes (comma-separated)
parser.add_argument(
    "--mlp_layers", 
    type=lambda s: [int(x) for x in s.split(',')],
    default=[256, 128, 64],
    help="Comma-separated list of MLP hidden layer dimensions (e.g., 256,128)"
)

# Optional: learning rate
parser.add_argument(
    "--lr", 
    type=float, 
    default=5e-2, 
    help="Learning rate (default: 5e-2)"
)

# Optional: temperature for contrastive loss
parser.add_argument(
    "--temperature", 
    type=float, 
    default=0.07, 
    help="Temperature for contrastive loss (default: 0.07)"
)

# Optional: weight decay
parser.add_argument(
    "--weight_decay", 
    type=float, 
    default=1e-4, 
    help="Weight decay for optimizer (default: 1e-4)"
)

parser.add_argument("--seed", type = int, default=None)

args = parser.parse_args()
if not os.path.exists(args.workdir):
    os.makedirs(args.workdir)
n_pos = args.pos
n_neg = args.neg

print("Finished parsing args")
print(args)

# Load the data =================================================
adata_st = sc.read_h5ad(args.source)
adata_histo = sc.read_h5ad(args.target)

def make_set(ad1, ad2, n_pos, n_neg, random_seed):
    embeddings_a=ad1.obsm[args.source_key]
    labels_a=ad1.obsm["brain_area_onehot"].toarray().nonzero()[-1]
    embeddings_b=ad2.obsm[args.target_key]
    labels_b=ad2.obsm["brain_area_onehot"].toarray().nonzero()[-1]
    
    return PairedContrastiveDataset(
        embeddings_a=embeddings_a, 
        labels_a=labels_a, 
        embeddings_b=embeddings_b, 
        labels_b=labels_b, 
        n_pos=n_pos,
        n_neg=n_neg,
        seed=random_seed
    )

train_dataset = make_set(
    adata_st[adata_st.obs.train_set],
    adata_histo[adata_histo.obs.train_set],
    n_pos=n_pos,
    n_neg=n_neg,
    random_seed=args.seed
)

# Use smaller batches for validation since it is smaller too
val_dataset = make_set(
    adata_st[adata_st.obs.val_set],
    adata_histo[adata_histo.obs.val_set],
    n_pos=max(1, int(n_pos/4)),
    n_neg=max(10, int(n_neg/4)),
    random_seed=args.seed
)

print(f"Train set: {len(train_dataset)}, Val set {len(val_dataset)}")

# Train the model ==========================================================

logger = pl.loggers.TensorBoardLogger(
    os.path.join(args.workdir, "logs"), name="SimCLR"
)
monitor_mode = "min" if ("loss" in args.monitor) else "max"
model = train_simclr(
    train_dataset, val_dataset, os.path.join(args.workdir, "simclr_ckpt"), limit_train_batches = 0.2, logger = logger, 
    refresh_rate = 5000, monitor = args.monitor, monitor_mode = monitor_mode, dropout = args.dropout,
    hidden_dims=args.mlp_layers[:-1], latent_dim = args.mlp_layers[-1],
    lr=args.lr, temperature=args.temperature, weight_decay=args.weight_decay,
    use_batchnorm = args.batchnorm, log_every = 1000, shuffle_every = 5, max_epochs = args.max_epochs
)

print ("Finished training SimCLR")

# Use the model to predict coordinates =====================================
model.eval()
batch_size = 500

st_dataloader = DataLoader(TensorDataset(
    torch.tensor(adata_st.obsm[args.source_key], dtype=torch.float32)
), batch_size = batch_size, shuffle=False, pin_memory=True)
st_embeddings = []

histo_dataloader = DataLoader(TensorDataset(
    torch.tensor(adata_histo.obsm[args.target_key], dtype=torch.float32)
), batch_size = batch_size, shuffle=False, pin_memory=True)
histo_embeddings = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

with torch.no_grad():
    for batch, in st_dataloader:
        emb = model.st_model.embed(batch.to(device))
        st_embeddings.append(emb.cpu())
    for batch, in histo_dataloader:
        emb = model.histo_model.embed(batch.to(device))
        histo_embeddings.append(emb.cpu())
    st_embeddings = torch.vstack(st_embeddings)
    histo_embeddings = torch.vstack(histo_embeddings)

# For now, use the full Histo set as train set for coordinate regression
train_embeddings = torch.vstack([histo_embeddings, st_embeddings[adata_st.obs.train_set]])
train_coordinates = np.vstack((
    adata_histo.obs.loc[:, ('x_st', 'y_st', 'z_st')].to_numpy(), 
    adata_st.obs.loc[adata_st.obs.train_set, ('x_st', 'y_st', 'z_st')].to_numpy()
))
val_embeddings = st_embeddings[adata_st.obs.val_set]
val_coordinates = adata_st.obs.loc[adata_st.obs.val_set, ('x_st', 'y_st', 'z_st')].to_numpy()
train_dataset = CoordinateDataset(train_embeddings, train_coordinates)
val_dataset = CoordinateDataset(val_embeddings, val_coordinates)

logger2 = pl.loggers.TensorBoardLogger(
    os.path.join(args.workdir, "logs"), name="Coordinate Regressor"
)

# Train the coordinate predictor
coord_model = train_coordinate_regressor(
    train_dataset, val_dataset, os.path.join(args.workdir, "reg_ckpts"), in_dim=train_embeddings.shape[-1], 
    batch_size = 256, max_epochs = 50, hidden_dims=[1024, 512], dropout = 0.4, logger = logger2
)