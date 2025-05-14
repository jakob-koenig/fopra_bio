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
from sklearn.metrics import classification_report, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

print("Devices", jax.devices())

# Solve the problem ================================================================================================
## Argument parsing
parser = argparse.ArgumentParser(description="Solve a translation problem.")

# Define arguments
parser.add_argument("-s", "--source", type=str, required=True, help="Path to the input source file")
parser.add_argument("-t", "--target", type=str, required=True, help="Path to the input target file")
parser.add_argument("-p", "--path", type=str, required=True, help="Path to the output")
parser.add_argument("-a", "--alpha", type=float, default = None, help="Alpha parameter (float)")
parser.add_argument("-e", "--epsilon", type=float, default = None, help="Epsilon parameter (float)")
parser.add_argument("-A", "--tau_a", type=float, default = 1.0, help="Tau a parameter (float)")
parser.add_argument("-B", "--tau_b", type=float, default = 1.0, help="Tau b parameter (float)")
parser.add_argument("-q", "--source_key", type=str, default = "pca_embedding", help="Key in source adata obsm")
parser.add_argument("-r", "--target_key", type=str, default = "uni_embedding", help="Key in target adata obsm")
parser.add_argument("--sample_target", action = "store_true")
parser.add_argument("--low_rank", action = "store_true", help = "Whether to decrease rank or split into fractions when encountering OOM.")
parser.add_argument("-l", "--linear_term", type=str, default = None)
parser.add_argument("--random_seed", type=int, default = None)

# Parse arguments
args = parser.parse_args()
path = args.path
adata_src = sc.read_h5ad(args.source)
adata_target = sc.read_h5ad(args.target)
alpha = args.alpha
epsilon = args.epsilon
source_key = args.source_key
target_key = args.target_key
linear_term = args.linear_term
tau_a = args.tau_a
tau_b = args.tau_b
fused = True if linear_term != None else False
if not fused:
    alpha = 1.0 if alpha == None else alpha
    epsilon = 1e-3 if epsilon == None else epsilon
else:
    alpha = 0.7 if alpha == None else alpha
    epsilon = 5e-3 if epsilon == None else epsilon
seed = args.random_seed
sample_target = args.sample_target
low_rank = args.low_rank
print("Arguments:")
print(args)

# Run the solver
# Downsample the target modality
if sample_target:
    adata_target = adata_target[adata_target.obs.in_sample]

print(f"Adata shapes: src {adata_src.shape} || target {adata_target.shape}")
# Split the data into fractions using the uniform sampling
def test_input_size(adata_src, adata_target, path, fractions = 1,  rank = -1):
    N = adata_src.shape[0]
    M = adata_target.shape[0]

    if fractions > 1:
        rng = np.random.default_rng(seed = seed)
        perm_N = rng.choice(N, size = N, replace = False)
        perm_M = rng.choice(M, size = M, replace = False)
        step_N = int(np.ceil(N/fractions))
        step_M = int(np.ceil(M/fractions))
        
        adatas_src = [adata_src[perm_N[i*step_N:(i+1)*step_N], :] for i in range(fractions)]
        adatas_target = [adata_target[perm_M[i*step_M:(i+1)*step_M], :] for i in range(fractions)]
    else:
        adatas_src = [adata_src, ]
        adatas_target = [adata_target, ]
        perm_n = np.arange(N)
        perm_M = np.arange(M)
        step_N = N,
        step_M = M
        
    print("Source Adatas:", [ad.shape for ad in adatas_src])
    print("Target Adatas:", [ad.shape for ad in adatas_target])

    for i in range(fractions):
        ad_src = adatas_src[i]
        ad_target = adatas_target[i]
        
        tp = TranslationProblem(adata_src=ad_src, adata_tgt=ad_target)
        tp = tp.prepare(src_attr=source_key, tgt_attr=target_key, joint_attr = linear_term)
        
        try:
            tp = tp.solve(alpha=alpha, epsilon=epsilon, rank = rank)
            tp.save(os.path.join(path, f"opt_transport{str(i)}.pkl"), overwrite = True)
            [x.delete() for x in jax.devices()[0].client.live_buffers()]
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(e)
                print("======================================FAILURE=========================================")
                return False  # OOM Error
            else:
                raise e  # Some other error
    # It worked :)
    if fractions > 1:
        np.save(os.path.join(path, f"permutation_source.npy"), perm_N)
        np.save(os.path.join(path, f"permutation_target.npy"), perm_M)
    return (adatas_src, adatas_target, N, M, step_N, step_M, perm_N, perm_M)

fractions = 1
rank = -1
max_fractions = 12
max_rank = 100
success = False
while (not success) and (fractions <= max_fractions) and (rank >= max_rank or rank == -1):  # stop at 12 fractions or rank less than 1000
    if low_rank:
        # Decrease rank
        rank = int(0.5 * rank) if rank > 0 else min(adata_target.shape[0], adata_src.shape[0])
        [x.delete() for x in jax.devices()[0].client.live_buffers()]
        gc.collect()
        print(f"===========================================Rank {rank}==========================================")
    else:
        # Increase number of splits
        fractions += 1
        print(f"===========================================Fractions {fractions}==========================================")
    success = test_input_size(adata_src, adata_target, path, fractions, rank)

if low_rank:
    print("Stopped at", "rank:", rank)
else:
    print("Stopped at", fractions, "fractions")
    
if fractions > 12 or (rank < 1000 and rank != -1):
    quit()

src_adatas, adatas, N, M, step_N, step_M, perm_N, perm_M = success

# Translate the problem =================================================================================================
def switch_to_cpu():
    """Switch JAX to CPU mode."""
    [x.delete() for x in jax.devices()[0].client.live_buffers()]
    gc.collect()
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    
switch_to_cpu()
# Reload the transitions problem
print(f"{fractions} fractions, with combined shape {(N, M)}")

tps = []
for i in range(fractions):    
    tp = TranslationProblem(adata_src=src_adatas[i], adata_tgt=adatas[i])
    tp = tp.prepare(src_attr=source_key, tgt_attr=target_key, joint_attr=linear_term)
    tp = tp.load(os.path.join(path, f"opt_transport{str(i)}.pkl"))
    tps.append(tp)
print("finished loading problems")

## Use custom translation method
def _get_features(
            adata,
            attr,
        ):
    data = getattr(adata, attr["attr"])
    key = attr.get("key")
    return data if key is None else data[key]

# Check if translation already happened
transformations = []
for p in range(fractions):
    print(f"======================================Fraction {p} ============================================")
    tp = tps[p]
    ad_target = adatas[p]
    src_attr = tp._src_attr
    tgt_attr = tp._tgt_attr
    features = _get_features(ad_target, attr=tgt_attr)

    batch_size = 10
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
    bt = np.concatenate(batch_transformed, axis=1)
    transformations.append(bt)

## Put it all together as one output array
transformed = np.concatenate(transformations, axis=0)

if fractions > 1:
    transformed = pd.DataFrame(transformed, index = perm_N).sort_index().to_numpy()
    adata_src.obs["fraction"] = -1
    adata_src.obs.iloc[perm_N, -1] = np.repeat(np.arange(fractions), step_N)[perm_N]
    adata_target.obs["fraction"] = -1
    adata_target.obs.iloc[perm_M, -1] = np.repeat(np.arange(fractions), step_M)[perm_M]

adata_src.obsm["transformed"] = transformed
np.save(os.path.join(path, "translated.npy"), transformed)

adata_src.write_h5ad(os.path.join(path, "adata_src.h5ad"))
adata_target.write_h5ad(os.path.join(path, "adata_target.h5ad"))

## Evaluate using plots and scores
# UMAP
print("Plotting UMAP")
# Make color map
set1 = set(list(adata_src.obs["brain_area"]))
set2 = set(list(adata_target.obs["brain_area"]))
cats = list(set1 | set2)
cmap = plt.get_cmap("tab20")
color_map = {cat: cmap(i / len(cats)) for i, cat in enumerate(cats)}

adata = sc.concat(
    [adata_src, adata_target],
    join="outer",
    label="batch",
    keys=["ST (translated)", "Histo"],
)
adata.obsm["X_translated"] = np.concatenate(
    (adata_src.obsm["transformed"], adata_target.obsm[target_key]), axis=0
)
sc.pp.neighbors(adata, use_rep="X_translated")
sc.tl.umap(adata)

fig_umap, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 13))

sc.pl.umap(adata, color=["batch"], ax=ax1, show=False)
ax1.legend()
ax1.set_title("Colored based on modality")

sc.pl.umap(adata, color=["brain_area"], ax=ax2, show=False)
ax2.set_title("Colored based on brain areas")

sc.pl.umap(adata, color=["fraction"], ax=ax3, show=False)
ax3.set_title("Colored based on fractions for GW solving")
plt.savefig(os.path.join(path, "umap_plot.pdf"), bbox_inches="tight", dpi=300)

# Create the cost plot
print("Plotting costs")
costs = [pd.DataFrame({"cost": p.solutions[('src', 'tgt')]._costs.copy()}) for p in tps]
costs = [costs[i].assign(converged=bool(tps[i].solutions[('src', 'tgt')]._output.converged))
         .assign(step = np.arange(len(costs[i]))).assign(problem=str(i))
    for i in range(len(costs))]
costs = pd.concat(costs, ignore_index = True)
costs = costs[costs.cost != -1]

cost_plot = (
    p9.ggplot(costs, p9.aes(x="step", y="cost", color="converged", group = "problem")) +
    p9.geom_line() +
    p9.labs(title = "Cost function during training")
)

p9.save_as_pdf_pages((cost_plot, ), os.path.join(path, "cost_plot.pdf"))

#Cell transitions
cell_transitions = []
for i in range(fractions):
    ad_src = src_adatas[i]
    ad_target = adatas[i]
    tp = tps[i]
    source_order = ad_src.obs["brain_area"].cat.categories
    target_order = ad_target.obs["brain_area"].cat.categories
    
    
    cell_transitions.append(tp.cell_transition(
        source="src",
        target="tgt",
        source_groups={"brain_area": source_order},
        target_groups={"brain_area": target_order},
        forward=True,
        key_added = None
    ))

plot_df = cell_transitions[0].reindex(index=set1, columns=set2, fill_value=0)
for df in cell_transitions[1:]:
    plot_df = plot_df.add(df.reindex(index=set1, columns=set2, fill_value=0), fill_value=0)
plot_df = plot_df / fractions

plt.figure(figsize=(10, 8))
sns.heatmap(plot_df, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)

# Add labels
plt.ylabel("ST brain area")
plt.xlabel("Histo brain area")
plt.title("Probability Heatmap for transitions")

# Show the plot
plt.savefig(os.path.join(path, "transitions_plot.pdf"), bbox_inches="tight", dpi=300)

print("Calculating scores")
def log_reg(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=500)
    )
    clf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    return(clf, y_test, y_pred)

clf, y_test, y_pred = log_reg(adata_target.obsm[target_key], adata_target.obs["brain_area"])
target_report = classification_report(y_test, y_pred)
src_report = classification_report(
    clf.predict(adata_src.obsm["transformed"]),
    adata_src.obs["brain_area"]
)
print("Brain area prediction performance on test set")
print(target_report)
print("Brain area prediction performance on translated source")
print(src_report)

# Location regression ============================================================================
def location_reg(X, df):

    y = df.loc[:, ["x_st", "y_st", "z_st"]].to_numpy()
    stratifier = df["brain_area"].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratifier, random_state=42)
    
    reg = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(max_depth = 15)
    )
    
    reg.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = reg.predict(X_test)
    return(reg, y_test, y_pred)

reg, reg_test, reg_pred = location_reg(adata_target.obsm[target_key], adata_target.obs)
target_r2 = r2_score(reg_test, reg_pred)
src_r2 = r2_score(
    reg.predict(adata_src.obsm["transformed"]),
    adata_src.obs.loc[:, ["x_st", "y_st", "z_st"]].to_numpy()
)
print("Location regression R2 on test set:", target_r2)
print("Location regression R2 on translation", src_r2)

# knn q===================================================================================================
def knn_metric(
    translation, source_features, target_features, target_ad, k = 1, emb_key = "uni_embedding", rank = True
):
    # calculate the k nearest neighbors on the translation
    target_embedding = target_ad.obsm[emb_key]
    source_features = np.array(source_features)
    target_features = np.array(target_features)
    N = source_features.shape[0]
    M = target_features.shape[0]
    
    # Use knn for faster computation
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(target_embedding)

    # Find top-k neighbors in target for each src sample
    distances, indices = nbrs.kneighbors(translation)

    # Calculate the MSE weighted by distance/rank of distance
    if rank:
        # Do not use aczual distances, just weight by rank
        distances = np.tile(np.arange(1, k + 1), (N, 1))
        # Add pseudocount to distances to avoid zero division, and turn into weights by taking inverse
        distances = 1/distances 
    else:
        distances = 1/(distances + 1e-8)

    big_translation = np.repeat(source_features[:, np.newaxis, :], k, axis=1)  # Make array to calculate error
    big_nbrs = target_features[indices]

    idwe = np.sum(np.sum(np.square(big_translation - big_nbrs), axis = 2) * distances, axis = 1)
    idwe = idwe / np.sum(distances, axis = 1)  # Normalize by weights

    # Compare to random by dividing by random distances
    rng = np.random.default_rng()
    perm = rng.choice(M, size = (N, k), replace = True)
    random_nbrs = target_features[perm]
    random_idwe = np.sum(np.sum(np.square(big_translation - random_nbrs), axis = 2) * distances, axis = 1)
    random_idwe = random_idwe / np.sum(distances, axis = 1)  # Normalize by weights

    idwe = 1 - np.mean(idwe) / np.mean(random_idwe)
    
    return float(np.mean(idwe))

idwe = knn_metric(
    adata_src.obsm["transformed"], 
    adata_src.obs.loc[:, ["x_st", "y_st", "z_st"]], 
    adata_target.obs.loc[:, ["x_st", "y_st", "z_st"]],
    adata_target, emb_key = target_key
)
print("kNN based coordinate interpolation score:", idwe)

def translation_metric(adatas_src, adatas_target, tps):
    predictions_T = []
    true_T = []
    transitions = [tp[('src', 'tgt')].solution.transport_matrix.__array__() for tp in tps]
    
    for i in range(fractions):
        ad_st = adatas_src[i]
        ad_histo = adatas_target[i]
        
        histo_coordinates = ad_histo.obs.loc[:, ["x_st", "y_st", "z_st"]].to_numpy()
        spatial_coordinates = ad_st.obs.loc[:, ["x_st", "y_st", "z_st"]].to_numpy()
        T = transitions[i]
        true_T.append(spatial_coordinates)
    
        for j in range(ad_st.shape[0]):
            t_j = T[j] / np.sum(T[j])
            t_j = np.repeat(t_j[:, None], 3, axis=1)
            prediction = np.sum(histo_coordinates * t_j, axis = 0)
            predictions_T.append(prediction)
    
    predictions_T = np.vstack(predictions_T)
    true_T = np.vstack(true_T)
    
    return r2_score(true_T, predictions_T)

translation_score = translation_metric(
    src_adatas, adatas, tps
)
print("translation metric based coordinate interpolation score:", translation_score)

with open(os.path.join(path, "scores.txt"), "w") as file:
    file.write("Brain area prediction performance on test set")
    file.write("\n")
    file.write(str(target_report))
    file.write("\n")
    file.write("Brain area prediction performance on translated source")
    file.write("\n")
    file.write(str(src_report))
    file.write("\n\n")

    file.write(f"Location regression R2 on test set: {target_r2} \n")
    file.write(f"Location regression R2 on translation {src_r2} \n\n")

    file.write(f"Coordinate interpolation score: {idwe} \n")
    file.write(f"Translation metric based coordinate interpolation score: {translation_score} \n")

    # Write out all the arguments
    file.write(f"path: {path}\n")
    file.write(f"source file: {args.source}\n")
    file.write(f"target file: {args.target}\n")
    file.write(f"alpha: {alpha}\n")
    file.write(f"epsilon: {epsilon}\n")
    file.write(f"source_key: {source_key}\n")
    file.write(f"target_key: {target_key}\n")
    file.write(f"linear_term: {linear_term}\n")
    file.write(f"tau_a: {tau_a}\n")
    file.write(f"tau_b: {tau_b}\n")
    file.write(f"fused: {fused}\n")
    file.write(f"random_seed: {seed}\n")
    file.write(f"sample_target: {sample_target}\n")
    file.write(f"low_rank: {low_rank}\n")

print("DONE")

