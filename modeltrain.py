"""
End-to-End Forgery Detection Training Script

This script performs training, validation, and testing of a dual-input model.
The model takes forensic features (4-channel input extracted from a .npz file stored in D:/final_dataset/)
and an RGB image (3-channel input extracted from a .npz file stored in D:/original_rgb/).
It outputs a segmentation mask for the manipulated area, a forgery type classification 
(copymove or splicing), and a binary classification (authentic or forged).

Pre-requisites:
 - PyTorch
 - torchvision
 - timm (for EfficientNet-B4): Install via pip install timm
 - Other common libraries: numpy, pandas, glob, etc.
 
Ensure that the directory structure is as follows:

D:/final_dataset/
    train/val/test/ (each with .npz files containing variables: tensor, mask, label, forgery_type)
D:/original_rgb/
    train/val/test/ (each with .npz files containing a tensor variable (3,512,512))
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm  # make sure timm is installed

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Directory Settings and Params
# =============================

FOR_FINAL_DATASET = r"D:/final_dataset1"   # Forensic .npz dataset (5 channels; train/val/test)
FOR_RGB_DATASET   = r"D:/original_rgb1"    # RGB .npz dataset (3 channels; train/val/test)
SAVE_MODEL_PATH   = "model.pt"             # Path to save the trained model

BATCH_SIZE        = 2      # Adjust as needed for GPU memory
NUM_EPOCHS        = 10     # Number of training epochs
LR                = 1e-4   # Learning rate


# =============================
# Dataset Class Definition
# =============================

class ForgeryDataset(Dataset):
    def __init__(self, forensic_dir, rgb_dir, split="train"):
        """
        :param forensic_dir: Path to D:/final_dataset/ (containing train/val/test folders)
        :param rgb_dir: Path to D:/original_rgb/ (containing train/val/test folders)
        :param split: one of ["train", "val", "test"]
        """
        self.forensic_path = os.path.join(forensic_dir, split)
        self.rgb_path      = os.path.join(rgb_dir, split)
        
        # Gather all .npz files in each path (assuming they match one-to-one by name)
        self.forensic_files = sorted(glob.glob(os.path.join(self.forensic_path, "*.npz")))
        self.rgb_files      = sorted(glob.glob(os.path.join(self.rgb_path, "*.npz")))
        
        # Ensure that both folders have the same number of files.
        assert len(self.forensic_files) == len(self.rgb_files), \
            f"Mismatch in # of files: {len(self.forensic_files)} vs {len(self.rgb_files)}"
        
    def __len__(self):
        return len(self.forensic_files)
    
    def __getitem__(self, idx):
        # Load forensic .npz file
        forensic_data = np.load(self.forensic_files[idx])
        # Expected shape: (5, 512, 512) where channels are:
        # [0]: normalized grayscale, [1:4]: enhanced RGB, [4]: mask (ground truth)
        forensic_tensor = forensic_data['tensor']  # (5, 512, 512)
        label           = forensic_data['label'].item()  # 0 or 1 for authentic/forged
        forgery_type    = str(forensic_data['forgery_type'])  # "copymove" or "splicing"
        
        # Prepare forensic input (first 4 channels)
        forensic_input = torch.from_numpy(forensic_tensor[:4]).float()  # (4,512,512)
        # For segmentation training, we use the last channel from the tensor.
        mask = torch.from_numpy(forensic_tensor[4]).float()   # (512,512)
        
        # Load the corresponding RGB .npz file for the same sample
        rgb_data = np.load(self.rgb_files[idx])
        rgb_input = rgb_data['tensor']  # expected shape: (3,512,512)
        rgb_input = torch.from_numpy(rgb_input).float()
        
        sample = {
            "forensic_input": forensic_input,  # (4,512,512)
            "rgb_input": rgb_input,            # (3,512,512)
            "mask": mask,                      # (512,512)
            "label": torch.tensor(label).long(),         # 0 or 1
            "forgery_type": forgery_type                    # "copymove" or "splicing"
        }
        return sample

        
# =============================
# Model Components
# =============================

# --- Forensic CNN Stream ---
class SimpleForensicCNN(nn.Module):
    def __init__(self, in_channels=4, out_channels=384):
        """
        A simple CNN that takes 4-channel forensic input and outputs a feature map.
        """
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Downsample (B,64,256,256)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: (B, 4, 512, 512)
        Returns a feature map, e.g. (B,384,256,256)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# --- RGB Stream using EfficientNet-B4 ---
class RGBEfficientNetB4(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=True, features_only=True)
    
    def forward(self, x):
        """
        x: (B,3,512,512)
        Returns a list of multi-scale feature maps.
        """
        features = self.backbone(x)
        return features  # e.g., returns list of 5 scales

# --- Feature Fusion ---
class SimpleFeatureFusion(nn.Module):
    def __init__(self, in_channels_e=384, in_channels_f=384, out_channels=512):
        """
        Fuses EfficientNet and Forensic CNN features.
        """
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels_e + in_channels_f, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self, eff_feat, for_feat):
        # Concatenate along channel dimension
        import pdb; pdb.set_trace()
        fused = torch.cat([eff_feat, for_feat], dim=1)
        fused = self.conv1x1(fused)
        fused = self.relu(fused)
        return fused  # (B, out_channels, H, W)

# --- Transformer Encoder ---
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=2048,
                                                   dropout=0.1,
                                                   activation='relu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: (B,512,H,W), expects H=W=16
        """
        B, C, H, W = x.shape
        # Flatten spatial dimensions to tokens: (B, H*W, C)
        x = x.reshape(B, C, H*W).permute(0, 2, 1)  # (B, 256, 512) if H=W=16
        x = self.transformer(x)  # (B,256,512)
        # Reshape back to (B,512,H,W)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

# --- Decoder (FPN-Style) and Output Heads ---
class SimpleFPNDecoder(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1)  # 16->32
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)           # 32->64
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)            # 64->128
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)             # 128->256
        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)             # 256->512
        
        # Mask head
        self.mask_head = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid   = nn.Sigmoid()
        
        # Classification heads: Global Average Pooling then a fully connected layer
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.forgery_type_fc = nn.Linear(16, 2)  # 2 classes: copymove/splicing
        self.binary_fc       = nn.Linear(16, 2)  # 2 classes: authentic/forged

    def forward(self, x):
        """
        x: (B,512,16,16)
        Returns: mask prediction, forgery type prediction, binary classification prediction.
        """
        x = F.relu(self.up1(x))  # (B,256,32,32)
        x = F.relu(self.up2(x))  # (B,128,64,64)
        x = F.relu(self.up3(x))  # (B,64,128,128)
        x = F.relu(self.up4(x))  # (B,32,256,256)
        x = F.relu(self.up5(x))  # (B,16,512,512)
        
        mask_pred = self.sigmoid(self.mask_head(x))  # (B,1,512,512)
        
        pooled = self.gap(x).squeeze(-1).squeeze(-1)   # (B,16)
        forgery_type_pred = self.forgery_type_fc(pooled)  # (B,2)
        binary_pred       = self.binary_fc(pooled)        # (B,2)
        
        return mask_pred, forgery_type_pred, binary_pred

# --- Complete Model ---
class ForgeryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.forensic_cnn = SimpleForensicCNN(in_channels=4, out_channels=384)
        self.rgb_stream   = RGBEfficientNetB4()
        
        # Fusion: We assume forensic stream and efficientnet produce 384-channel features.
        self.fusion = SimpleFeatureFusion(in_channels_e=384, in_channels_f=384, out_channels=512)
        self.transformer = SimpleTransformerEncoder(d_model=512, nhead=8, num_layers=4)
        self.decoder     = SimpleFPNDecoder(in_channels=512)
        
    def forward(self, forensic_input, rgb_input):
        """
        forensic_input: (B,4,512,512)
        rgb_input: (B,3,512,512)
        """
        # 1) Forensic CNN feature extraction; output shape (B,384,256,256)
        f_feat = self.forensic_cnn(forensic_input)
        
        # 2) EfficientNet-B4 feature extraction; we select one scale.
        rgb_feats = self.rgb_stream(rgb_input)
        import pdb; pdb.set_trace()
        # Assuming we choose a feature map with shape (B,384,16,16)
        # If the backbone outputs a different channel number, adjust accordingly.
        e_feat = rgb_feats[-2]  # Selecting second last scale for example.
        
        # 3) Resize forensic features to match e_feat resolution (assume 16x16)
        f_feat_small = F.interpolate(f_feat, size=e_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # 4) Fuse the EfficientNet feature and forensic feature
        fused_feat = self.fusion(e_feat, f_feat_small)  # (B,512,16,16)
        
        # 5) Transformer Encoder
        transformed = self.transformer(fused_feat)      # (B,512,16,16)
        
        # 6) Decoder (FPN-style) to obtain mask and classifications
        mask_pred, forgery_type_pred, binary_pred = self.decoder(transformed)
        
        return mask_pred, forgery_type_pred, binary_pred

        
# =============================
# Training, Validation, and Testing Functions
# =============================

def train_and_validate():
    # Prepare datasets
    from torch.utils.data import DataLoader
    train_ds = ForgeryDataset(FOR_FINAL_DATASET, FOR_RGB_DATASET, split="train")
    val_ds   = ForgeryDataset(FOR_FINAL_DATASET, FOR_RGB_DATASET, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = ForgeryModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Losses: BCE for segmentation, CrossEntropy for classification heads.
    bce = nn.BCELoss()               
    ce  = nn.CrossEntropyLoss()      
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            forensic_input = batch["forensic_input"].to(DEVICE)
            rgb_input      = batch["rgb_input"].to(DEVICE)
            mask_gt        = batch["mask"].to(DEVICE)
            label          = batch["label"].to(DEVICE)  # binary: 0 or 1
            forgery_type   = batch["forgery_type"]          # string labels: "copymove" or "splicing"
            
            # Convert forgery_type to integer: 0 for copymove, 1 for splicing
            ft_labels = []
            for ft in forgery_type:
                ft_labels.append(0 if ft == "copymove" else 1)
            ft_labels = torch.tensor(ft_labels, dtype=torch.long, device=DEVICE)
            
            optimizer.zero_grad()
            mask_pred, ft_pred, binary_pred = model(forensic_input, rgb_input)
            
            # Adjust mask shape for BCE loss
            mask_pred = mask_pred.squeeze(1)  # (B,512,512)
            loss_mask = bce(mask_pred, mask_gt)
            loss_ft = ce(ft_pred, ft_labels)
            loss_bin = ce(binary_pred, label)
            loss = loss_mask + loss_ft + loss_bin
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                forensic_input = batch["forensic_input"].to(DEVICE)
                rgb_input      = batch["rgb_input"].to(DEVICE)
                mask_gt        = batch["mask"].to(DEVICE)
                label          = batch["label"].to(DEVICE)
                forgery_type   = batch["forgery_type"]
                
                ft_labels = []
                for ft in forgery_type:
                    ft_labels.append(0 if ft == "copymove" else 1)
                ft_labels = torch.tensor(ft_labels, dtype=torch.long, device=DEVICE)
                
                mask_pred, ft_pred, binary_pred = model(forensic_input, rgb_input)
                mask_pred = mask_pred.squeeze(1)
                loss_mask = bce(mask_pred, mask_gt)
                loss_ft   = ce(ft_pred, ft_labels)
                loss_bin  = ce(binary_pred, label)
                loss_v    = loss_mask + loss_ft + loss_bin
                val_loss += loss_v.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"Model saved to {SAVE_MODEL_PATH}")
    return model

def test_model(model_path=SAVE_MODEL_PATH):
    test_ds = ForgeryDataset(FOR_FINAL_DATASET, FOR_RGB_DATASET, split="test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = ForgeryModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_ft_correct = 0
    total_bin_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            forensic_input = batch["forensic_input"].to(DEVICE)
            rgb_input      = batch["rgb_input"].to(DEVICE)
            mask_gt        = batch["mask"].to(DEVICE)
            label          = batch["label"].to(DEVICE)
            forgery_type   = batch["forgery_type"]
            
            ft_labels = []
            for ft in forgery_type:
                ft_labels.append(0 if ft == "copymove" else 1)
            ft_labels = torch.tensor(ft_labels, dtype=torch.long, device=DEVICE)
            
            mask_pred, ft_pred, binary_pred = model(forensic_input, rgb_input)
            mask_pred = mask_pred.squeeze(1)
            loss_mask = bce(mask_pred, mask_gt)
            loss_ft   = ce(ft_pred, ft_labels)
            loss_bin  = ce(binary_pred, label)
            loss      = loss_mask + loss_ft + loss_bin
            total_loss += loss.item()
            
            ft_pred_class = torch.argmax(ft_pred, dim=1)
            total_ft_correct += (ft_pred_class == ft_labels).sum().item()
            
            bin_pred_class = torch.argmax(binary_pred, dim=1)
            total_bin_correct += (bin_pred_class == label).sum().item()
            
            total_samples += label.size(0)
    
    avg_loss = total_loss / len(test_loader)
    ft_acc   = total_ft_correct / total_samples
    bin_acc  = total_bin_correct / total_samples
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Forgery Type Accuracy: {ft_acc*100:.2f}%")
    print(f"Binary Authentic/Forged Accuracy: {bin_acc*100:.2f}%")

def run_inference_single_image(forensic_npz_path, rgb_npz_path, model_path=SAVE_MODEL_PATH):
    # Load forensic .npz and prepare forensic input
    forensic_data = np.load(forensic_npz_path)
    forensic_tensor = forensic_data['tensor']  # (5,512,512)
    forensic_input = torch.from_numpy(forensic_tensor[:4]).unsqueeze(0).float().to(DEVICE)
    
    # Load corresponding RGB .npz
    rgb_data = np.load(rgb_npz_path)
    rgb_input = torch.from_numpy(rgb_data['tensor']).unsqueeze(0).float().to(DEVICE)
    
    model = ForgeryModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    with torch.no_grad():
        mask_pred, ft_pred, bin_pred = model(forensic_input, rgb_input)
    
    mask_pred  = (mask_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    ft_class   = torch.argmax(ft_pred, dim=1).item()  # 0: copymove, 1: splicing
    bin_class  = torch.argmax(bin_pred, dim=1).item()   # 0: authentic, 1: forged
    
    ft_str = "copymove" if ft_class == 0 else "splicing"
    bin_str = "authentic" if bin_class == 0 else "forged"
    
    print("Inference Results:")
    print("Mask shape:", mask_pred.shape)
    print("Forgery Type:", ft_str)
    print("Binary Classification:", bin_str)

# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    # Train and validate the model
    model = train_and_validate()
    
    # Test the model
    test_model()
    
    # Optionally, run inference on a single image (update paths as needed)
    # run_inference_single_image("path_to_forensic_file.npz", "path_to_rgb_file.npz")
