import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"]="cpu"

from moscot.problems.cross_modality import TranslationProblem

import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
import argparse

import matplotlib.pyplot as plt
import plotnine as p9

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

import scanpy as sc
import jax

print("Devices", jax.devices())

## Argument parsing
parser = argparse.ArgumentParser(description="Solve a translation problem.")

# Define arguments
parser.add_argument("-p", "--path", type=str, required=True, help="Path to the input files")
parser.add_argument("-t", "--target", type=str, required=True, help="Path to the input target file")
parser.add_argument("-s", "--source", type=str, required=True, help="Path to the source file")
parser.add_argument("-l", "--linear_term", type=str, default = None)
parser.add_argument("-q", "--source_key", type=str, default = "pca_embedding", help="Key in source adata obsm")
parser.add_argument("-r", "--target_key", type=str, default = "uni_embedding", help="Key in target adata obsm")
parser.add_argument("--sample_target", action = "store_true")

args = parser.parse_args()
print(args)
path = args.path
adata_src = sc.read_h5ad(args.source)
adata_target = sc.read_h5ad(args.target)
source_key = args.source_key
target_key = args.target_key
linear_term = args.linear_term
sample_target = args.sample_target

## Load problems
fractions = len([file for file in os.listdir(path) if file.startswith("opt_transport")])
if sample_target:
    adata_target = adata_target[adata_target.obs.in_sample]

N = adata_src.shape[0]
M = adata_target.shape[0]

print(f"{fractions} fractions, with combined shape {(N, M)}")
if fractions > 1:
    perm_N = np.load(os.path.join(path, "permutation_source.npy"))
    perm_M = np.load(os.path.join(path, "permutation_target.npy"))
    step_N = int(np.ceil(N/fractions))
    step_M = int(np.ceil(M/fractions))

tps = []
adatas = []
src_adatas = []
for i in range(fractions):
    if fractions > 1:
        ad_src = adata_src[perm_N[i*step_N:(i+1)*step_N], :]
        ad_target = adata_target[perm_M[i*step_M:(i+1)*step_M], :]
    else:
        ad_src = adata_src
        ad_target = adata_target
    
    tp = TranslationProblem(adata_src=ad_src, adata_tgt=ad_target)
    tp = tp.prepare(src_attr=source_key, tgt_attr=target_key, joint_attr=linear_term)
    tp = tp.load(os.path.join(path, f"opt_transport{str(i)}.pkl"))
    tps.append(tp)
    adatas.append(ad_target)
    src_adatas.append(ad_src)
print("finished loading problems")

## Translate the problems
def _get_features(
            adata,
            attr,
        ):
    data = getattr(adata, attr["attr"])
    key = attr.get("key")
    return data if key is None else data[key]

# Check if tranlation already happened
if not "translated.npy" in os.listdir(path):
    transformations = []
    for p in range(fractions):
        print(f"======================================Fraction {p} ============================================")
        tp = tps[p]
        ad_target = adatas[p]
        src_attr = tp._src_attr
        tgt_attr = tp._tgt_attr
        features = _get_features(ad_target, attr=tgt_attr)
    
        batch_size = 100
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
    transformed = pd.DataFrame(transformed, index = perm_N).sort_index().to_numpy()
    np.save(os.path.join(path, "translated.npy"), transformed)
else:
    transformed = np.load(os.path.join(path, "translated.npy"))

adata_src.obsm["transformed"] = transformed
adata_src.obs["fraction"] = -1
adata_src.obs.iloc[perm_N, -1] = np.repeat(np.arange(fractions), step_N)[perm_N]
adata_target.obs["fraction"] = -1
adata_target.obs.iloc[perm_M, -1] = np.repeat(np.arange(fractions), step_M)[perm_M]

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
def location_reg(X, df, alpha = 1.0):

    y = df.loc[:, ["x_st", "y_st", "z_st"]].to_numpy()
    stratifier = df["brain_area"].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratifier, random_state=42)
    
    reg = make_pipeline(
        StandardScaler(),
        RandomForestRegressor()
    )
    
    reg.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = reg.predict(X_test)
    return(reg, y_test, y_pred)

reg, reg_test, reg_pred = location_reg(adata_target.obsm[target_key], adata_target.obs, alpha = 500.0)
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
    color_by_fraction = []
    transitions = [tp[('src', 'tgt')].solution.transport_matrix.__array__() for tp in tps]
    
    for i in range(fractions):
        ad_st = adatas_src[i]
        ad_histo = adatas_target[i]
        
        histo_coordinates = ad_histo.obs.loc[:, ["x_st", "y_st", "z_st"]].to_numpy()
        spatial_coordinates = ad_st.obs.loc[:, ["x_st", "y_st", "z_st"]].to_numpy()
        T = transitions[i]
        true_T.append(spatial_coordinates)
    
        for j in range(ad_st.shape[0]):
            color_by_fraction.append(i)
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

print("DONE")


