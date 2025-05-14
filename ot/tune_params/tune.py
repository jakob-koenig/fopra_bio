from moscot.problems.cross_modality import TranslationProblem

import numpy as np

import matplotlib.pyplot as plt

import anndata as ad
import scanpy as sc
import jax
import os
import argparse
import importlib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real, Categorical
import gc
import datetime

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Torch using device: {device}")

print("Jax Devices:", jax.devices())

## Argument parsing
parser = argparse.ArgumentParser(description="Solve a translation problem.")

## Argument parsing ==============================================================================
parser = argparse.ArgumentParser(description="Solve a translation problem.")

# Define arguments
parser.add_argument("-s", "--source", type=str, required=True, help="Path to the input source file")
parser.add_argument("-t", "--target", type=str, required=True, help="Path to the input target file")
parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output")
parser.add_argument("-q", "--source_key", type=str, default = "pca_embedding", help="Key in source adata obsm")
parser.add_argument("-r", "--target_key", type=str, default = "uni_embedding", help="Key in target adata obsm")
parser.add_argument("-l", "--linear_term", type=str, default = None)
parser.add_argument("-m", "--metric", type=str, default = "MLP")
parser.add_argument("--random_seed", type=int, default = None)
parser.add_argument("--imbalanced", action="store_true", help = "Use only one slide from the spatial transcriptomics data")
parser.add_argument("--try_metric_spaces", action="store_true", help = "Try out all different metric spaces in solve")
parser.add_argument("-c", "--cost", type=str, default = "sq_euclidean")

args = parser.parse_args()
out_path = args.output
adata_src = sc.read_h5ad(args.source)
adata_target = sc.read_h5ad(args.target)

source_key = args.source_key
target_key = args.target_key
linear_term = args.linear_term
fused = False if linear_term == None else True
seed = args.random_seed
metric = args.metric
imbalanced = args.imbalanced
try_metric_spaces = args.try_metric_spaces
cost = args.cost

print("Arguments:")
print(args)
print("adata_src.obsm contains")
print([key for key in adata_src.obsm])
print("adata_target.obsm contains")
print([key for key in adata_target.obsm])

# Sample adatas to 12000
N = adata_src.shape[0]
M = adata_target.shape[0]

rng = np.random.default_rng(seed = seed)

# Split into training data for MLP
X, y = adata_target.obsm[target_key], adata_target.obs.loc[:, ["x_st", "y_st", "z_st"]].to_numpy()
stratifier = adata_target.obs["brain_area"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=stratifier, random_state=seed)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
print("NA in Train set:", np.isnan(X_train).any(), np.isnan(y_train).any())
print("NA in Val set:", np.isnan(X_val).any(), np.isnan(y_val).any())

if "adata_src.h5py" in os.listdir(out_path) and "adata_target.h5py" in os.listdir(out_path):
    adata_src = sc.read_h5ad(os.path.join(out_path, "adata_src.h5py"))
    adata_target = sc.read_h5ad(os.path.join(out_path, "adata_target.h5py"))
elif not imbalanced:
    sample_size = 30000
    perm_N = rng.choice(N, size = N, replace = False)
    perm_M = rng.choice(M, size = M, replace = False)
    adata_src = adata_src[perm_N[0:sample_size], :]
    adata_target = adata_target[perm_M[0:sample_size], :]
    adata_src.write_h5ad(os.path.join(out_path, "adata_src.h5py"))
    adata_target.write_h5ad(os.path.join(out_path, "adata_target.h5py"))
else:
    print("Imbalanced problem, only using slide Zhuang-ABCA-1.092")
    adata_src = adata_src[adata_src.obs.brain_section_label == "Zhuang-ABCA-1.092"]  # Pick only one slide (631)
    perm_M = rng.choice(M, size = M, replace = False)
    adata_target = adata_target[perm_M[0:12500], :]
    adata_src.write_h5ad(os.path.join(out_path, "adata_src.h5py"))
    adata_target.write_h5ad(os.path.join(out_path, "adata_target.h5py"))
    
print(f"Adata shapes: {adata_src.shape} || {adata_target.shape}")
src_coords = adata_src.obs.loc[:, ["x_st", "y_st", "z_st"]].to_numpy()
target_coords = adata_target.obs.loc[:, ["x_st", "y_st", "z_st"]].to_numpy()
print("NA in src_coords:", np.isnan(src_coords).any())

## Train an MLP to use as regressor ==============================================================================
class MLP(nn.Module):
    def __init__(self, input_size=368, hidden1=1024, hidden2=256, output_size=3):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_size)  # Output (x, y, z)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model and move it to GPU
model = MLP(input_size = adata_target.shape[1])

if not "mlp.pth" in os.listdir(out_path) and (metric == "MLP"):
    model = model.to(device)
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Move data to GPU
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.float32).to(device)
    
    # Training loop with early stopping
    epochs = 100
    batch_size = 256
    patience = 5  # Stop if no improvement for 5 epochs
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    print("training MLP")
    
    for epoch in range(epochs):
        model.train()  # Set to training mode
        permutation = torch.randperm(X_train.size(0))  # Shuffle data
    
        # Training step
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]
    
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            if torch.isnan(loss):
                print("Loss is NaN!")
            else:
                loss.backward()
                optimizer.step()
    
        # Validation step
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, y_val).item()
    
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")
    
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1
    
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break  # Stop training

    model.eval()
    model = model.to("cpu")
    torch.save(model.state_dict(), os.path.join(out_path, "mlp.pth"))
elif metric == "MLP":
    model.load_state_dict(torch.load(os.path.join(out_path, "mlp.pth"), map_location="cpu"))
    model.eval()
    model = model.to("cpu")

def evaluate_mlp(translation):
    with torch.no_grad():
        predictions = torch.tensor(scaler.transform(translation), dtype=torch.float32)
        predictions = model(predictions)
        predictions = torch.nan_to_num(predictions).numpy()
    return r2_score(src_coords, predictions)

# Use translation metric for easier benchmark
def knn_interpolation(
    translation, source_features, target_features, target_ad, emb_key = "uni_embedding"
):
    # calculate the k nearest neighbors on the translation
    target_embedding = target_ad.obsm[emb_key]
    source_features = np.array(source_features)
    target_features = np.array(target_features)
    N = source_features.shape[0]
    M = target_features.shape[0]
    
    # Use knn for faster computation
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_embedding)

    # Find top-k neighbors in target for each src sample
    _, indices = nbrs.kneighbors(translation)
    nbr_features = target_features[indices.flatten()]

    idwe = np.sum(np.square(source_features - nbr_features), axis = 1)

    # Compare to random by dividing by random distances
    rng = np.random.default_rng()
    perm = rng.choice(M, size = (N), replace = True)
    random_nbrs = target_features[perm]
    random_idwe = np.sum(np.square(source_features - random_nbrs), axis = 1)

    idwe = 1 - np.mean(idwe) / np.mean(random_idwe)
    
    return float(np.mean(idwe))

def evaluate_idwe(translation):
    return knn_interpolation(
        translation,
        src_coords, 
        target_coords,
        adata_target, emb_key = target_key
    )

def translation_metric(tp):
    predictions_T = [] 
    T = tp[('src', 'tgt')].solution.transport_matrix.__array__()
    if np.isnan(T.flatten()).any():
        return -1e6  # Bad performance to indicate sth went wrong in the solution step

    for j in range(adata_src.shape[0]):
        t_j = T[j] / np.sum(T[j])
        t_j = np.repeat(t_j[:, None], 3, axis=1)
        prediction = np.sum(target_coords * t_j, axis = 0)
        predictions_T.append(prediction)
    
    predictions_T = np.vstack(predictions_T)
    if np.isnan(predictions_T.flatten()).any():
        return -1e6

    return r2_score(src_coords, predictions_T)

use_translation_metric = False
if (metric == "MLP"):
    evaluate_translation = evaluate_mlp
elif (metric == "IDWE"):
    evaluate_translation = evaluate_idwe
elif (metric == "translation"):
    use_translation_metric = True
else:
    raise ValueError("Unknown evaluation metric")

## Define the objective function ==============================================================================
def switch_to_cpu():
    """Switch JAX to CPU mode."""
    [x.delete() for x in jax.devices()[0].client.live_buffers()]
    gc.collect()
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
    #os.environ["JAX_PLATFORMS"] = "cpu"      # Force CPU execution

def switch_to_gpu():
    """Switch JAX to GPU mode."""
    gc.collect()
    jax.config.update("jax_default_device", jax.devices("gpu")[0])
    #os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # Enable GPU
    #os.environ["JAX_PLATFORMS"] = "gpu"           # Force GPU execution

def _get_features(
            adata,
            attr,
        ):
    data = getattr(adata, attr["attr"])
    key = attr.get("key")
    return data if key is None else data[key]

def objective(alpha, epsilon, tau_a, tau_b, source_key, target_key, linear_term):
    # Solve the problem on the GPU
    tp = TranslationProblem(adata_src=adata_src, adata_tgt=adata_target)
    print("Source, Target:", (source_key, target_key))
    tp = tp.prepare(src_attr=source_key, tgt_attr=target_key, joint_attr=linear_term, cost=cost)
    
    tp = tp.solve(alpha=alpha, epsilon=epsilon, tau_a=tau_a, tau_b=tau_b)
    if not use_translation_metric:
        tp.save(os.path.join(out_path, "tmp.pkl"), overwrite = True)
        del tp
    
        switch_to_cpu()
        print("Translating Problem", datetime.datetime.now().time())
        tp = TranslationProblem(adata_src=adata_src, adata_tgt=adata_target)
        tp = tp.prepare(src_attr=source_key, tgt_attr=target_key, joint_attr=linear_term, cost=cost)
        tp = tp.load(os.path.join(out_path, "tmp.pkl"))
    
        #Translate the problem on CPU
        src_attr = tp._src_attr
        tgt_attr = tp._tgt_attr
        features = _get_features(adata_target, attr=tgt_attr)
    
        batch_size = 120
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
        print("\n Finished translation", datetime.datetime.now().time())
        
        translation = np.concatenate(batch_transformed, axis=1)
        if np.isnan(translation).any():
            print("NAN in translation: ", np.isnan(translation).sum())
        translation = np.nan_to_num(translation)
        score = evaluate_translation(translation)
        # Clean up
        del tp, batch_transformed, translation, features, prop
    else:
        score = translation_metric(tp)

    return score

## Do a hyperparameter search =============================================================================
# Wrapper function for optimization
def objective_wrapper(params):
    if fused:
        items = 4
        epsilon, tau_a, tau_b, alpha = params[:items]
    else:
        items = 3
        epsilon, tau_a, tau_b = params[:items]
        alpha = 1.0

    if try_metric_spaces:
        source_key_, target_key_, linear_term_ = params[items:]
    else:
        # Use global defaults
        source_key_ = source_key
        target_key_ = target_key
        linear_term_ = linear_term
        
    print(f"Starting job with {alpha},{epsilon},{tau_a},{tau_b}, {source_key}, {target_key}, {linear_term}")
    
    score = objective(alpha, epsilon, tau_a, tau_b, source_key_, target_key_, linear_term_)
    print(score)
    with open(os.path.join(out_path, "param_search.txt"), "a") as file:
        file.write(f"{alpha},{epsilon},{tau_a},{tau_b},{score},{source_key_},{target_key_},{linear_term_}\n")
    score = -score  # We expect to mnimize, so we need to invert the scores
    return score

# Define the search space for parameters
search_space = [
    Real(1e-6, 0.1, name='epsilon'),   # Entropic regularization strength
    Real(0.1, 1.0, name='tau_a'),     # Unbalanced OT parameter for source
    Real(0.1, 1.0, name='tau_b')      # Unbalanced OT parameter for target
]
if fused:
    search_space.append(Real(0.01, 0.95, name='alpha'))      # Balance parameter (0 = pure GW, 1 = pure Wasserstein
if try_metric_spaces:
    search_space += [
        Categorical(['pca_embedding', 'pca_plus_slides', 'pca_plus_slides_scaled'], name = "source_key"),  # Source metric for GW
        Categorical(["uni_pca_95", "uni_pca_plus_coords"], name = "target_key"),  # Target metric for GW
        Categorical(["brain_area_onehot", "brain_area_similarities"], name = "linear_term")  # Linear term for wasserstein distance
    ]

print(f"alpha,epsilon,tau_a,tau_b,r2,source_key, target_key, linear_term")
with open(os.path.join(out_path, "param_search.txt"), "w") as file:
        file.write(f"alpha,epsilon,tau_a,tau_b,{metric},source_key,target_key,linear_term\n")
# Run Bayesian Optimization (Gaussian Process)
res = gp_minimize(objective_wrapper, search_space, n_calls=100, random_state=seed)

# Best parameters found
if fused:
    items = 4
    best_epsilon, best_tau_a, best_tau_b, best_alpha = res.x[:items]
else:
    items = 3
    best_epsilon, best_tau_a, best_tau_b = res.x[:items]
    best_alpha = 1.0

if try_metric_spaces:
    best_source_key, best_target_key, best_linear_term = res.x[items:]
else:
    best_source_key = source_key
    best_target_key = target_key
    best_linear_term = linear_term
    
best_cost = -res.fun

# Print results
print(f"Optimal Parameters:")
print(f"  α (alpha): {best_alpha}")
print(f"  ε (epsilon): {best_epsilon}")
print(f"  τₐ (tau_a): {best_tau_a}")
print(f"  τᵦ (tau_b): {best_tau_b}")
print(f"  source key: {best_source_key}")
print(f"  target key: {best_target_key}")
print(f"  linear term: {best_linear_term}")
print(f"Maximum {metric} Achieved: {best_cost}")
