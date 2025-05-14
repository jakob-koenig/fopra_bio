from moscot.problems.cross_modality import TranslationProblem

import numpy as np

import matplotlib.pyplot as plt

import anndata as ad
import scanpy as sc
import jax
import os
from tqdm import tqdm
import argparse

print("Devices", jax.devices())

## Argument parsing
parser = argparse.ArgumentParser(description="Solve a translation problem.")

# Define arguments
parser.add_argument("-s", "--source", type=str, required=True, help="Path to the input source file")
parser.add_argument("-t", "--target", type=str, required=True, help="Path to the input target file")
parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output")
parser.add_argument("-a", "--alpha", type=float, default = None, help="Alpha parameter (float)")
parser.add_argument("-e", "--epsilon", type=float, default = None, help="Epsilon parameter (float)")
parser.add_argument("-q", "--source_key", type=str, default = "pca_embedding", help="Key in source adata obsm")
parser.add_argument("-r", "--target_key", type=str, default = "uni_embedding", help="Key in target adata obsm")
parser.add_argument("--sample_target", action = "store_true")
parser.add_argument("-l", "--linear_term", type=str, default = None)
parser.add_argument("--random_seed", type=int, default = None)

# Parse arguments
args = parser.parse_args()
out_path = args.output
adata_src = sc.read_h5ad(args.source)
adata_target = sc.read_h5ad(args.target)
alpha = args.alpha
epsilon = args.epsilon
source_key = args.source_key
target_key = args.target_key
linear_term = args.linear_term
fused = True if linear_term != None else False
if not fused:
    alpha = 1.0 if alpha == None else alpha
    epsilon = 1e-3 if epsilon == None else epsilon
else:
    alpha = 0.7 if alpha == None else alpha
    epsilon = 5e-3 if epsilon == None else epsilon
seed = args.random_seed
sample_target = args.sample_target 
print("Arguments:")
print(args)

# Run the solver
# Downsample the target modality
if sample_target:
    adata_target = adata_target[adata_target.obs.in_sample]

print(f"Adata shapes: src {adata_src.shape} || target {adata_target.shape}")
# Split the data into fractions using the uniform sampling
def test_input_size(fractions, adata_src, adata_target, out_path):
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
    print("Source Adatas:", [ad.shape for ad in adatas_src])
    print("Target Adatas:", [ad.shape for ad in adatas_target])

    for i in range(fractions):
        ad_src = adatas_src[i]
        ad_target = adatas_target[i]
        
        tp = TranslationProblem(adata_src=ad_src, adata_tgt=ad_target)
        tp = tp.prepare(src_attr=source_key, tgt_attr=target_key, joint_attr = linear_term)
        
        try:
            tp = tp.solve(alpha=alpha, epsilon=epsilon)
            tp.save(os.path.join(out_path, f"opt_transport{str(i)}.pkl"), overwrite = True)
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
        np.save(os.path.join(out_path, f"permutation_source.npy"), perm_N)
        np.save(os.path.join(out_path, f"permutation_target.npy"), perm_M)
    return True

fractions = 0
success = False
while (not success) and (fractions <= 12):  # stop at 12
    fractions += 1
    print(f"===========================================Fractions {fractions}==========================================")
    success = test_input_size(fractions, adata_src, adata_target, out_path)

print("Stopped at", fractions, "fractions")