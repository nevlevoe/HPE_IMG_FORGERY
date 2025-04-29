import os
import cv2
import numpy as np
import glob
import random
import pickle
from tqdm import tqdm
import shutil
import pandas as pd

# ============
# SETTINGS
# ============

# --- Base directories (change as needed) ---
base_preproc_path = r"C:\Users\msada"
download_folder = r"Downloads"  # folder that contains the blank authentic mask

# Copymove preprocessed paths
copymove_enhanced_dir = os.path.join(base_preproc_path, "copymove_preprocessed", "DEFACTO_ELA_Enhanced")
copymove_normalized_dir = os.path.join(base_preproc_path, "copymove_preprocessed", "DEFACTO_ELA_Normalized")
copymove_mask_dir = os.path.join(base_preproc_path, "copymove_preprocessed", "mask")

# Splicing preprocessed paths
splicing_enhanced_dir = os.path.join(base_preproc_path, "splicing_preprocessed", "DEFACTO_ELA_Enhanced")
splicing_normalized_dir = os.path.join(base_preproc_path, "splicing_preprocessed", "DEFACTO_ELA_Normalized")
splicing_mask_dir = os.path.join(base_preproc_path, "splicing_preprocessed", "mask")

# For authentic images, the mask is a blank image (all black)
authentic_mask_path = os.path.join(download_folder, "authentic_blank.jpg")

# Output dataset folder
final_dataset_dir = r"D:\final_dataset"
train_dir = os.path.join(final_dataset_dir, "train")
val_dir   = os.path.join(final_dataset_dir, "val")
test_dir  = os.path.join(final_dataset_dir, "test")

# Create output directories (delete if exists)
for d in [train_dir, val_dir, test_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# Desired split ratios
train_ratio = 0.8
val_ratio   = 0.1
test_ratio  = 0.1

# ============
# UTILITY FUNCTIONS
# ============

def load_image_as_rgb(path):
    """Load an image from path with cv2, convert BGR -> RGB."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Image not found or failed to load: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_image_as_gray(path):
    """Load an image as grayscale."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or failed to load: {path}")
    return img

def build_tensor(enhanced_path, normalized_path, mask_path):
    """
    Build a 5-channel tensor for one image entity.
    Returns a dictionary with keys: 'tensor', 'mask'
    where tensor is an array of shape (5, H, W).

    The channels are:
       - 1 channel: the normalized image (grayscale)
       - 3 channels: the enhanced image in RGB (loaded from the _enhanced_ file)
       - 1 channel: the mask image (grayscale)
    """
    # Load the normalized image (grayscale) and ensure it is 512x512 (as preprocessed)
    norm = load_image_as_gray(normalized_path)  # shape: (H, W)
    
    # Load the enhanced image as color (RGB)
    enhanced = load_image_as_rgb(enhanced_path)  # (H, W, 3)
    
    # Load the mask as grayscale. For authentic images, this will be the blank mask.
    mask = load_image_as_gray(mask_path)  # (H, W)
    
    # Check that all images have same spatial dimensions. (Assumed 512x512)
    H, W = norm.shape
    
    # Convert images to float32 (or as desired for model input), and normalize if needed.
    norm_float  = norm.astype(np.float32) / 255.0
    enhanced = enhanced.astype(np.float32) / 255.0
    mask_float = mask.astype(np.float32) / 255.0

    # Transpose/expand to channel-first format:
    # Normalized image: (1, H, W)
    norm_channel = np.expand_dims(norm_float, axis=0)     
    # Enhanced image: (3, H, W)
    enhanced = np.transpose(enhanced, (2, 0, 1))           
    # Mask: (1, H, W)
    mask_channel = np.expand_dims(mask_float, axis=0)       

    # Concatenate in the order: [norm (1), enhanced (3), mask (1)]
    tensor = np.concatenate([norm_channel, enhanced, mask_channel], axis=0)  # (5, H, W)
    
    return tensor, mask.astype(np.float32)  # mask as label image

def process_single_sample(sample):
    """
    Given a sample descriptor (a dict with keys: 'enhanced_path', 'normalized_path', 
    'mask_path', 'label', 'forgery_type', and 'filename'),
    load the images and build the tensor dictionary.
    """
    try:
        tensor, mask = build_tensor(sample['enhanced_path'], sample['normalized_path'], sample['mask_path'])
        sample_dict = {
            'tensor': tensor,
            'mask': mask,  # shape (H,W)
            'label': sample['label'],  # 0=authentic, 1=forged
            'forgery_type': sample['forgery_type']  # 'copymove' or 'splicing'
        }
        return sample['filename'], sample_dict
    except Exception as e:
        print(f"Error processing {sample['filename']}: {e}")
        return None

# ============
# GATHERING FILES & SAMPLING
# ============

def get_samples(enhanced_dir, normalized_dir, mask_dir, label, forgery_type, mask_for_authentic=None):
    """
    Build a list of sample descriptors from a given dataset type. Assumes that for each image,
    the filename (without extension) is the same in the enhanced and normalized folders.
    
    If mask_for_authentic is provided (for authentic, e.g., a single blank mask), use that.
    """
    samples = []
    # List all image files in the enhanced directory (assuming *.png for enhanced)
    # Adjust glob pattern if needed.
    pattern = os.path.join(enhanced_dir, "*")
    for enhanced_path in glob.glob(pattern):
        filename = os.path.basename(enhanced_path)
        # Derive normalized path from the same filename (with jpg for normalized, as stated)
        normalized_path = os.path.join(normalized_dir, filename)
        # if file extensions differ, try replacing extension if needed:
        if not os.path.exists(normalized_path):
            name, ext = os.path.splitext(filename)
            # try .jpg for normalized images
            normalized_path = os.path.join(normalized_dir, name + ".jpg")
            if not os.path.exists(normalized_path):
                print(f"Normalized file not found for {enhanced_path}")
                continue
        
        # For forged samples, use the mask from mask_dir (assuming same filename and .jpg)
        if label == 1:
            mask_path = os.path.join(mask_dir, filename)
            if not os.path.exists(mask_path):
                name, ext = os.path.splitext(filename)
                mask_path = os.path.join(mask_dir, name + ".jpg")
                if not os.path.exists(mask_path):
                    print(f"Mask file not found for {enhanced_path}")
                    continue
        else:
            # For authentic images, use the provided blank mask
            mask_path = mask_for_authentic
        
        sample = {
            'filename': f"{forgery_type}_{filename}",
            'enhanced_path': enhanced_path,
            'normalized_path': normalized_path,
            'mask_path': mask_path,
            'label': label,  # 0: authentic, 1: forged
            'forgery_type': forgery_type
        }
        samples.append(sample)
    return samples

# Gather samples for both copymove and splicing:
copymove_forged_samples = get_samples(
    enhanced_dir = os.path.join(copymove_enhanced_dir, "forged"),
    normalized_dir = os.path.join(copymove_normalized_dir, "forged"),
    mask_dir = copymove_mask_dir,
    label = 1,
    forgery_type = "copymove"
)

print(copymove_forged_samples[0])

copymove_authentic_samples = get_samples(
    enhanced_dir = os.path.join(copymove_enhanced_dir, "authentic"),
    normalized_dir = os.path.join(copymove_normalized_dir, "authentic"),
    mask_dir = None,  # will use blank mask
    label = 0,
    forgery_type = "copymove",
    mask_for_authentic = authentic_mask_path
)

splicing_forged_samples = get_samples(
    enhanced_dir = os.path.join(splicing_enhanced_dir, "forged"),
    normalized_dir = os.path.join(splicing_normalized_dir, "forged"),
    mask_dir = splicing_mask_dir,
    label = 1,
    forgery_type = "splicing"
)

splicing_authentic_samples = get_samples(
    enhanced_dir = os.path.join(splicing_enhanced_dir, "authentic"),
    normalized_dir = os.path.join(splicing_normalized_dir, "authentic"),
    mask_dir = None,  # will use blank mask
    label = 0,
    forgery_type = "splicing",
    mask_for_authentic = authentic_mask_path
)

#save

copymove_forged_samples.sort(key=lambda x: x['filename'])
copymove_authentic_samples.sort(key=lambda x: x['filename'])
dfc = pd.read_csv("Downloads//copymove.csv")
# Compute split indices based on overall count
num_total = len(copymove_forged_samples)
num_train = int(train_ratio * num_total)
num_val = int(val_ratio * num_total)
# The rest will be for test
num_test = num_total - num_train - num_val

for idx, sample in enumerate(tqdm(copymove_forged_samples, desc="Processing and Saving copymove samples")):
    if dfc['copymove'][idx] == 1:
        result = process_single_sample(sample)
        filename, data = result
    elif dfc['copymove'][idx] == 0:
        result = process_single_sample(copymove_authentic_samples[idx])
        filename, data = result
        
    # Determine the split for saving based on the sample index.
    if idx < num_train:
        out_dir = train_dir
    elif idx < num_train + num_val:
        out_dir = val_dir
    else:
        out_dir = test_dir
    
    out_path = os.path.join(out_dir, filename.rstrip(".png") + ".npz")
    np.savez_compressed(out_path,
                        tensor=data['tensor'],
                        mask=data['mask'],
                        label=data['label'],
                        forgery_type=data['forgery_type'])

splicing_forged_samples.sort(key=lambda x: x['filename'])
splicing_authentic_samples.sort(key=lambda x: x['filename'])
dfs = pd.read_csv("Downloads//splicing.csv")
# Compute split indices based on overall count
num_total = len(splicing_forged_samples)
num_train = int(train_ratio * num_total)
num_val = int(val_ratio * num_total)
# The rest will be for test
num_test = num_total - num_train - num_val

for idx, sample in enumerate(tqdm(splicing_forged_samples, desc="Processing and Saving splicing samples")):
    if dfs['splicing'][idx] == 1:
        result = process_single_sample(sample)
        filename, data = result
    elif dfs['splicing'][idx] == 0:
        result = process_single_sample(splicing_authentic_samples[idx])
        filename, data = result
        
    # Determine the split for saving based on the sample index.
    if idx < num_train:
        out_dir = train_dir
    elif idx < num_train + num_val:
        out_dir = val_dir
    else:
        out_dir = test_dir
    
    out_path = os.path.join(out_dir, filename.rstrip(".png") + ".npz")
    np.savez_compressed(out_path,
                        tensor=data['tensor'],
                        mask=data['mask'],
                        label=data['label'],
                        forgery_type=data['forgery_type'])
print("Dataset saved successfully in", final_dataset_dir)
