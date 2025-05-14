import timm
import torch
import os
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import ast
import argparse
import sys

# Parse args
parser = argparse.ArgumentParser(prog='Dataloader', description='Create dataloaders for training')
parser.add_argument("output_folder")
parser.add_argument("image_folder")
parser.add_argument("patch_size")

args = parser.parse_args()
print(args)
if not os.path.isdir(args.output_folder):
    print(f"Output directory <{args.output_folder}> not found.")
    sys.exit(1)
patch_size = int(args.patch_size)
image_path = args.image_folder
batch_size = 1

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model", device)
timm_kwargs = {
   'model_name': 'vit_giant_patch14_224',
   'img_size': 224, 
   'patch_size': 14, 
   'depth': 24,
   'num_heads': 24,
   'init_values': 1e-5, 
   'embed_dim': 1536,
   'mlp_ratio': 2.66667*2,
   'num_classes': 0, 
   'no_embed_class': True,
   'mlp_layer': timm.layers.SwiGLUPacked, 
   'act_layer': torch.nn.SiLU, 
   'reg_tokens': 8, 
   'dynamic_img_size': True
  }
local_dir = "/p/project1/hai_fzj_bda/koenig8/histo/assets/ckpts/uni2-h"
model = timm.create_model(**timm_kwargs)
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=device), strict=True)
model = model.to(device)
model.eval()

# Load the patching info
csv_path = "/p/project1/hai_fzj_bda/berr1/Data/full_data.csv"
df = pd.read_csv(csv_path).drop(columns=('Unnamed: 0'))

# Function to patch the images
def make_patches(pixel_coords, img, patch_size = 224):
    patches_sample = []
    half_size = patch_size // 2  # Half the patch size (112)
    padding_count = 0
    
    for centroid in pixel_coords:
        centroid = ast.literal_eval(centroid)
        x = centroid["center_x"]
        y = centroid["center_y"]
    
        x_start, x_end = max(x - half_size, 0), min(x + half_size, img.shape[1])
        y_start, y_end = max(y - half_size, 0), min(y + half_size, img.shape[0])
    
        # Extract patch and pad if necessary
        patch = img[y_start:y_end, x_start:x_end]
        if patch.shape != (patch_size, patch_size, 3):
            padding_count += 1
            pad_x = patch_size - (x_end - x_start)
            pad_y = patch_size - (y_end - y_start)
            patch = np.pad(patch, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')
        patches_sample.append(patch)
        
    patches_sample = np.array(patches_sample)
    if padding_count > 0:
        print("\t padded", padding_count, "out of", len(pixel_coords))
    return patches_sample

# Patch the images
transform = transforms.Compose(
 [
  transforms.Resize(224),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
 ]
)

# Forward pass of the model
def uni_forward(X, device = "cuda"):
    N = X.shape[0]
    features = []
    
    for i in range(0, N):
        batch = transform(Image.fromarray(X[i])).unsqueeze(dim=0).to(device=device)
        with torch.inference_mode():
            features.append(model(batch).cpu())

    features = torch.cat(features, dim = 0).detach().cpu().numpy()
    return(features)

completed_files = [file.split(".")[0] for file in os.listdir(args.output_folder) if not(file.startswith("."))]

for img_path in os.listdir(image_path):
    if img_path.startswith("."):
        continue

    img_file = img_path.split(".")[0]
    if img_file in completed_files:
        print("Skipping", img_file, "....")
        continue
        
    img = np.array(Image.open(os.path.join(image_path, img_path)))
    df_img = df[df.image_id.str.startswith(img_file)]
    if len(df_img) == 0:
        print("No annotations for", img_file, "....skipping.....")
        continue

    print("File", img_file)
    patches = make_patches(df_img["pixel_coord"], img, patch_size)
    N = patches.shape[0]

    print("Patches.shape:", patches.shape)
    if not((patches.shape[1] == patch_size) & (patches.shape[2] == patch_size) & (patches.shape[3] == 3)):
        print("\t Skipped due to wrong dimensions....")
        continue

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Frees unused memory
        torch.cuda.ipc_collect()

    # Use UNI
    features = uni_forward(patches, device = device)

    # Save in output folder
    np.save(os.path.join(args.output_folder, ".".join([img_file, "npy"])), features)
    
    
    