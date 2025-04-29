# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 19:04:38 2025

@author: msada
"""

import os
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
import pandas as pd
import tqdm
import numpy as np

# 1. Input paths for copy-move and splicing datasets
copy_move_forged_folder = r"C:\Users\msada\OneDrive\Documents\HPE_info\sampled_copymove_dataset_final\copymove_img"
copy_move_authentic_folder = r"C:\Users\msada\OneDrive\Documents\HPE_info\sampled_copymove_dataset_final\Authentic_Images"
splicing_forged_folder = r"C:\Users\msada\OneDrive\Documents\HPE_info\sampled_splicing_dataset_final\sampled_img\img"
splicing_authentic_folder = r"C:\Users\msada\OneDrive\Documents\HPE_info\sampled_splicing_dataset_final\sampled_img\Authentic Images"

# Output root directory 
final_dataset_dir = r"D:/original_rgb"

train_dir = os.path.join(final_dataset_dir, "train")
val_dir   = os.path.join(final_dataset_dir, "val")
test_dir  = os.path.join(final_dataset_dir, "test")

# Ensure output directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 2. Function to resize an image to 512x512, convert to RGB, and convert to a tensor (3x512x512)
def process_image(image_path):
    """
    Load an image, resize it to 512x512, convert to RGB, and convert it to a tensor.
    Returns a tensor of shape 3x512x512.
    """
    # Define transform sequence
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to 512x512
        transforms.ToTensor()           # Convert to tensor and scales pixel values to [0,1]
    ])
    
    # Open image with Pillow and ensure it is in RGB mode
    img = Image.open(image_path).convert('RGB')
    tensor_img = transform(img)  # tensor shape will be [3, 512, 512]
    return tensor_img

# 3. Gather file names from input directories and add respective suffixes
import os

def get_image_files(folder, prefix):
    """
    Returns a list of renamed file names from a folder.
    Converts '0_000000000321.tif' to '<prefix>_0_000000000321.tif'
    """
    files = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.tif')):
            base, ext = os.path.splitext(filename)
            new_filename = f"{prefix}_{base}{ext}"
            files.append(new_filename)
    return files



# Retrieve file lists from all four input directories
copy_move_forged_files = get_image_files(copy_move_forged_folder, "copymove")
copy_move_authentic_files = get_image_files(copy_move_authentic_folder, "copymove")
splicing_forged_files = get_image_files(splicing_forged_folder, "splicing")
splicing_authentic_files = get_image_files(splicing_authentic_folder, "splicing")

copy_move_forged_files.sort()  # Sorts in-place alphabetically (case-sensitive)
copy_move_authentic_files.sort()  # Sorts in-place alphabetically (case-sensitive)
splicing_forged_files.sort()  # Sorts in-place alphabetically (case-sensitive)
splicing_authentic_files.sort()  # Sorts in-place alphabetically (case-sensitive)

print("this is the name in list:",copy_move_forged_files[0])
#ratios
train_ratio = 0.8
val_ratio = 0.1

dfc = pd.read_csv("Downloads//copymove.csv")
# Compute split indices based on overall count
num_total = len(copy_move_forged_files)
num_train = int(train_ratio * num_total)
num_val = int(val_ratio * num_total)
# The rest will be for test
num_test = num_total - num_train - num_val

for idx, sample in enumerate(tqdm.tqdm(copy_move_forged_files, desc="Processing and Saving copymove samples")):
    if dfc['copymove'][idx] == 1:
        # Use slicing instead of lstrip to remove fixed prefix
        sample = sample[len("copymove_"):]
        data = process_image(os.path.join(copy_move_forged_folder, sample))
    elif dfc['copymove'][idx] == 0:
        auth_sample = copy_move_authentic_files[idx][len("copymove_"):]
        data = process_image(os.path.join(copy_move_authentic_folder, auth_sample))
        
    # Determine the split for saving based on the sample index.
    if idx < num_train:
        out_dir = train_dir
    elif idx < num_train + num_val:
        out_dir = val_dir
    else:
        out_dir = test_dir
    if dfc['copymove'][idx] == 1:
        out_path = os.path.join(out_dir, "copymove_" + sample + ".npz")
    elif dfc['copymove'][idx] == 0:
        out_path = os.path.join(out_dir, "copymove_" + auth_sample + ".npz")
    # Convert tensor to numpy array before saving
    np.savez_compressed(out_path, tensor=data.numpy())
    print(f"[{idx}] Saved: {os.path.basename(out_path)} → in → {out_dir}")

dfs = pd.read_csv("Downloads//splicing.csv")
# Compute split indices based on overall count
num_total = len(splicing_forged_files)
num_train = int(train_ratio * num_total)
num_val = int(val_ratio * num_total)
# The rest will be for test
num_test = num_total - num_train - num_val

for idx, sample in enumerate(tqdm.tqdm(splicing_forged_files, desc="Processing and Saving splicing samples")):
    # Note: Using 'dfs' for splicing related information
    if dfs['splicing'][idx] == 1:
        sample = sample[len("splicing_"):]
        data = process_image(os.path.join(splicing_forged_folder, sample))
    elif dfs['splicing'][idx] == 0:
        auth_sample = splicing_authentic_files[idx][len("splicing_"):]
        data = process_image(os.path.join(splicing_authentic_folder, auth_sample))
        
    # Determine the split for saving based on the sample index.
    if idx < num_train:
        out_dir = train_dir
    elif idx < num_train + num_val:
        out_dir = val_dir
    else:
        out_dir = test_dir
    if dfs['splicing'][idx] == 1:
        out_path = os.path.join(out_dir, "splicing_" + sample + ".npz")
    elif dfs['splicing'][idx] == 0:
        out_path = os.path.join(out_dir, "splicing_" + auth_sample + ".npz")
    # Save the tensor after converting to numpy
    np.savez_compressed(out_path, tensor=data.numpy())
    print(f"[{idx}] Saved: {os.path.basename(out_path)} → in → {out_dir}")
print("all things sampled properly")