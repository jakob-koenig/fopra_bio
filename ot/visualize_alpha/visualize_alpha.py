from moscot.problems.cross_modality import TranslationProblem

import numpy as np

import matplotlib.pyplot as plt

import anndata as ad
import scanpy as sc
import jax
import os
import argparse
import gc

import pandas as pd
from sklearn import preprocessing as pp
import plotnine as p9

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, r2_score, classification_report
from sklearn.pipeline import make_pipeline
import time

print("Devices", jax.devices())

# Solve the problem ================================================================================================
## Argument parsing
parser = argparse.ArgumentParser(description="Solve a translation problem.")

# Define arguments
parser.add_argument("-s", "--source", type=str, required=True, help="Path to the input source file")
parser.add_argument("-t", "--target", type=str, required=True, help="Path to the input target file")
parser.add_argument("-p", "--path", type=str, required=True, help="Path to the output")
parser.add_argument("-q", "--source_key", type=str, default = "pca_embedding", help="Key in source adata obsm")
parser.add_argument("-r", "--target_key", type=str, default = "uni_pca_95", help="Key in target adata obsm")
parser.add_argument("-l", "--linear_term", type=str, default = "brain_area_onehot")
parser.add_argument("--random_seed", type=int, default = 0)
parser.add_argument("--translate", action="store_true", help="Translate the problem?")

# Parse arguments
args = parser.parse_args()
path = args.path
adata_src = sc.read_h5ad(args.source)
adata_target = sc.read_h5ad(args.target)

source_key = args.source_key
target_key = args.target_key
linear_term = args.linear_term

alpha = 0.13
seed = args.random_seed
print("Arguments:")
print(args)

sample_size = 25000
rng = np.random.default_rng(seed = seed)
adata_target = adata_target[adata_target.obs.in_sample]  # Downsample the target modality

N = adata_src.shape[0]
M = adata_target.shape[0]
perm_N = rng.choice(N, size = N, replace = False)
perm_M = rng.choice(M, size = M, replace = False)
adata_src = adata_src[perm_N[0:sample_size], :]
adata_target = adata_target[perm_M[0:sample_size], :]
source_order = adata_src.obs["brain_area"].cat.categories
target_order = adata_target.obs["brain_area"].cat.categories

# Timer for computation speeds ==============================================================0
class Timer:
    def __init__(self):
        self.last_time = time.time()

    def check(self, label=""):
        now = time.time()
        elapsed = now - self.last_time
        print(f"[{label}] Time since last: {elapsed:.3f} s")
        self.last_time = now

timer = Timer()

# Run OT ===============================================================================================================

def switch_to_cpu():
    """Switch JAX to CPU mode."""
    [x.delete() for x in jax.devices()[0].client.live_buffers()]
    gc.collect()
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    print("Switched JAX to CPU mode.")

def switch_to_gpu():
    """Switch JAX to GPU mode."""
    gc.collect()
    jax.config.update("jax_default_device", jax.devices("gpu")[0])
    print("Switched JAX to GPU mode.")

def _get_features(
            adata,
            attr,
        ):
    data = getattr(adata, attr["attr"])
    key = attr.get("key")
    return data if key is None else data[key]

def log_reg(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=500)
    )
    clf.fit(X_train, y_train)
    
    return(clf, X_train, X_test, y_train, y_test)

if "adata_out.h5py" in os.listdir(path):
    print("reusing adata")
    adata = sc.read_h5ad(os.path.join(path, "adata_out.h5py"))
    assert (adata.obs["brain_area"].to_numpy() == 
        np.hstack([adata_src.obs["brain_area"], adata_target.obs["brain_area"]])).all() 
else:
    adata = sc.concat(
        [adata_src, adata_target],
        join="outer",
        label="batch",
        keys=["ST (translated)", "Histo"],
    )
    
print("Adata shape:", adata.shape)
    

clf, X_train, X_test, y_train, y_test = log_reg(adata_target.obsm[target_key], adata_target.obs["brain_area"])
y_pred = clf.predict(X_test)
target_f1 = f1_score(y_test, y_pred, average = "weighted")
print("Histology brain area classification:", classification_report(y_test, y_pred))

if not "adata_out.h5py" in os.listdir(path):
    adata.uns["brain_area_target_f1"] = target_f1
    adata.uns["brain_area_scores"] = {}
    adata.uns["translation_scores"] = {}
    adata.uns["converged"] = {}
    adata.uns["translation_matrices"] = {}
    adata.uns["cell_transitions"] = {}

def translation_metric(tp):
    predictions_T = []
    T = tp[('src', 'tgt')].solution.transport_matrix.__array__()
        
    src_coordinates = adata_src.obs.loc[:, ["x_st", "y_st", "z_st"]].to_numpy()
    target_coordinates = adata_target.obs.loc[:, ["x_st", "y_st", "z_st"]].to_numpy()

    for j in range(adata_src.shape[0]):
        t_j = T[j] / np.sum(T[j])
        t_j = np.repeat(t_j[:, None], 3, axis=1)
        prediction = np.sum(target_coordinates * t_j, axis = 0)
        predictions_T.append(prediction)
    
    predictions_T = np.vstack(predictions_T)
    return r2_score(src_coordinates, predictions_T)

timer.check("Loading data")
print("Beginning the solving")
counter = -1
for epsilon in np.logspace(-4, -0.2, num=30):  
    counter += 1
    if f"{epsilon:.6f}" in adata.uns["translation_scores"].keys():
        print("Skipping epsilon:", epsilon)
        continue
        
    print(f"epsilon: {epsilon}")
    # Solve
    print("Solving")
    tp = TranslationProblem(adata_src=adata_src, adata_tgt=adata_target)
    tp = tp.prepare(src_attr=source_key, tgt_attr=target_key, joint_attr=linear_term)
    tp = tp.solve(alpha=alpha, epsilon=epsilon)
    tp.save(os.path.join(path, f"tp{counter}.pkl"), overwrite = True)
    timer.check("Solving")

    if args.translate:
        # Translate
        del tp
    
        switch_to_cpu()
        tp = TranslationProblem(adata_src=adata_src, adata_tgt=adata_target)
        tp = tp.prepare(src_attr=source_key, tgt_attr=target_key, joint_attr=linear_term)
        tp = tp.load(os.path.join(path, f"tp{counter}.pkl"))

    
        #Translate the problem on CPU
        src_attr = tp._src_attr
        tgt_attr = tp._tgt_attr
        features = _get_features(adata_target, attr=tgt_attr)
    
        batch_size = 90
        batch_transformed = []
        prop = tp["src", "tgt"]

        print("Performing", int(features.shape[1] / batch_size) + 1, "translation steps")
        [print("_", end='') for _ in range(int(features.shape[1] / batch_size) + 1)]
        print("")
        for i in range(0, features.shape[1], batch_size):
            print('■', end='')
            batch_transformed.append(prop.pull(
                features[:, i:i+batch_size], subset = None, normalize = False, scale_by_marginals = True
            ))
        print("")
        
        transformed = np.concatenate(batch_transformed, axis=1)
        if np.isnan(transformed).any():
            print("NAN in translation: ", np.isnan(transformed).sum())
        transformed = np.nan_to_num(transformed)
    
        adata_src.obsm["transformed"] = transformed
        np.save(os.path.join(path, "translated.npy"), transformed)
        timer.check("Translating")
        
    sol = tp.solutions[('src', 'tgt')]._output
    
    converged = bool(sol.converged)
    adata.uns["converged"][f"{epsilon:.6f}"] = converged
    transport_matrix = np.array(sol.matrix)
    adata.uns["translation_matrices"][f"{epsilon:.6f}"] = transport_matrix

    # Get the transitions
    print("Cell transitions plot")
    
    cell_transition = tp.cell_transition(
        source="src",
        target="tgt",
        source_groups={"brain_area": source_order},
        target_groups={"brain_area": target_order},
        forward=True,
        key_added=None
    )
    adata.uns["cell_transitions"][f"{epsilon:.6f}"] = cell_transition
    timer.check("Cell Transitions")

    # Predict coordinates
    print("Predicting coordinates")
    trans_score = translation_metric(tp)
    print(f"Translation based coordinate score: {trans_score:.6f}")
    adata.uns["translation_scores"][f"{epsilon:.6f}"] =  trans_score
    adata.write_h5ad(os.path.join(path, "adata_out.h5py"))
    timer.check("Coordinates")

    ## Evaluate using plots and scores ==================================================
    if args.translate:
        print("Plotting UMAP")
        adata.obsm["X_translated"] = np.concatenate(
            (adata_src.obsm["transformed"], adata_target.obsm[target_key]), axis=0
        )
        sc.pp.neighbors(adata, use_rep="X_translated")
        sc.tl.umap(adata)
        adata.obsm[f"{epsilon:.6f}_umap"] = adata.obsm['X_umap'].copy()
        timer.check("UMAP")

        # Predict brain areas
        print("Predicting areas")
        src_f1 = f1_score(
            adata_src.obs["brain_area"],
            clf.predict(adata_src.obsm["transformed"]),
            average = "weighted"
        )
        adata.uns["brain_area_scores"][f"{epsilon:.6f}"] = src_f1
        print(f"brain area score: {src_f1:.6f}")
        timer.check("Brain areas")
        switch_to_gpu()
    else:
        switch_to_cpu(), switch_to_gpu()  # To clear memory

print("DONE")

