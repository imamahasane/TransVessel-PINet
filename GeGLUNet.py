import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from skimage import io, exposure
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch.nn.functional as F
from PIL import Image
from skimage import exposure

from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import directed_hausdorff

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Dataset (unchanged)
class DRIVEDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=64, train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.train = train
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        
    def __len__(self):
        return len(self.image_files) * 20  # Generate 20 patches per image
    
    def __getitem__(self, idx):
        img_idx = idx // 20
        img_path = os.path.join(self.image_dir, self.image_files[img_idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[img_idx])
        
        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        image = (image - image.min()) / (image.max() - image.min())
        image = exposure.equalize_adapthist(image)
        image = (image - image.min()) / (image.max() - image.min())
        mask = (mask > 0).astype(np.float32)
        
        if self.train:
            h, w = image.shape
            i = np.random.randint(0, h - self.patch_size)
            j = np.random.randint(0, w - self.patch_size)
            image = image[i:i+self.patch_size, j:j+self.patch_size]
            mask = mask[i:i+self.patch_size, j:j+self.patch_size]
        
        return torch.from_numpy(image).unsqueeze(0).float(), torch.from_numpy(mask).unsqueeze(0).float()

# GeGLU Layer
class GeGLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels * 2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x_proj = self.proj(x)
        x, gate = x_proj.chunk(2, dim=1)
        return self.bn(x * torch.sigmoid(gate))

# Attention Gate (unchanged)
class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels):
        super().__init__()
        self.W_g = nn.Conv2d(gating_channels, in_channels, 1)
        self.W_x = nn.Conv2d(in_channels, in_channels, 1)
        self.psi = nn.Conv2d(in_channels, 1, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, g):
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        psi = self.relu(g_conv + x_conv)
        psi = self.sigmoid(self.psi(psi))
        return x * psi

class GeGLUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Encoder with GeGLU
        self.enc1 = GeGLU(in_channels, 64)
        self.enc2 = GeGLU(64, 128)
        self.enc3 = GeGLU(128, 256)
        self.enc4 = GeGLU(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = GeGLU(512, 1024)
        
        # Decoder with Attention Gates
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att4 = AttentionGate(512, 512)
        self.dec4 = GeGLU(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att3 = AttentionGate(256, 256)
        self.dec3 = GeGLU(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2 = AttentionGate(128, 128)
        self.dec2 = GeGLU(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att1 = AttentionGate(64, 64)
        self.dec1 = GeGLU(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(e4))
        
        # Decoder with Attention
        d4 = self.up4(bottleneck)
        a4 = self.att4(e4, d4)
        d4 = torch.cat([a4, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        a3 = self.att3(e3, d3)
        d3 = torch.cat([a3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        a2 = self.att2(e2, d2)
        d2 = torch.cat([a2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        a1 = self.att1(e1, d1)
        d1 = torch.cat([a1, d1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.out(d1)), bottleneck

# Hybrid Loss (unchanged)

class EdgeAwareLoss(nn.Module):
    def __init__(self, edge_weight=0.5):
        super().__init__()
        self.edge_weight = edge_weight
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred, target):
        # Register Sobel kernels to the device
        self.sobel_kernel_x = self.sobel_kernel_x.to(pred.device)
        self.sobel_kernel_y = self.sobel_kernel_y.to(pred.device)
        
        # Compute edges for prediction and target
        pred_edges_x = F.conv2d(pred, self.sobel_kernel_x, padding=1)
        pred_edges_y = F.conv2d(pred, self.sobel_kernel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + 1e-6)
        
        target_edges_x = F.conv2d(target, self.sobel_kernel_x, padding=1)
        target_edges_y = F.conv2d(target, self.sobel_kernel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + 1e-6)
        
        # Normalize edges to [0, 1]
        pred_edges = (pred_edges - pred_edges.min()) / (pred_edges.max() - pred_edges.min() + 1e-6)
        target_edges = (target_edges - target_edges.min()) / (target_edges.max() - target_edges.min() + 1e-6)
        
        # Edge-aware binary cross-entropy
        edge_loss = F.binary_cross_entropy(pred_edges, target_edges)
        
        return self.edge_weight * edge_loss


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice_weight = 1.0
        self.jaccard_weight = 0.2
        self.lpcl_weight = 0.3
        self.edge_weight = 0.5  # New weight for edge loss
        self.edge_loss = EdgeAwareLoss(edge_weight=self.edge_weight)
    
    def dice_loss(self, pred, target):
        smooth = 1.0
        intersection = (pred * target).sum()
        return 1.0 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    def jaccard_loss(self, pred, target):
        smooth = 1.0
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return 1.0 - (intersection + smooth) / (union + smooth)
    
    def lpcl_loss(self, features, target):
        batch_size = features.size(0)
        features = features.view(batch_size, -1)
        features = F.normalize(features, dim=1)
        target = target.view(batch_size, -1).mean(dim=1)
        target = (target > 0.5).float()
        mask = target.unsqueeze(0) == target.unsqueeze(1)
        mask = mask.float()
        similarity = torch.mm(features, features.T)
        pos_pairs = similarity * mask
        neg_pairs = similarity * (1 - mask)
        num_pos = mask.sum() - batch_size
        num_neg = (1 - mask).sum()
        pos_loss = -pos_pairs.sum() / num_pos if num_pos > 0 else torch.tensor(0.0, device=features.device)
        neg_loss = neg_pairs.sum() / num_neg if num_neg > 0 else torch.tensor(0.0, device=features.device)
        return pos_loss + neg_loss
    
    def forward(self, pred, target, features):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        jaccard = self.jaccard_loss(pred, target)
        lpcl = self.lpcl_loss(features, target)
        edge = self.edge_loss(pred, target)  # New edge loss
        return bce + self.dice_weight * dice + self.jaccard_weight * jaccard + self.lpcl_weight * lpcl + edge


image_dir = "/Users/imamahasan/MyData/Code/Retinal_Vessel_S/Models/AttUKAN/test/images"
mask_dir = "/Users/imamahasan/MyData/Code/Retinal_Vessel_S/Models/AttUKAN/test/mask"
train_dataset = DRIVEDataset(image_dir, mask_dir, train=True)
train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GeGLUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = HybridLoss()

def calculate_accuracy(pred, target):
    pred = (pred > 0.5).float()
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

def calculate_dice(pred, target):
    smooth = 1.0
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

for epoch in range(100):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_dice = 0.0
    
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs, features = model(images)
        
        # Calculate metrics
        acc = calculate_accuracy(outputs, masks)
        dice = calculate_dice(outputs, masks)
        
        loss = criterion(outputs, masks, features)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_acc += acc
        running_dice += dice
    
    # Calculate epoch averages
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_acc / len(train_loader)
    epoch_dice = running_dice / len(train_loader)
    
    print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Dice: {epoch_dice:.4f}")

device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model = GeGLUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = HybridLoss()

def calculate_metrics(pred, target):
    pred_bin = (pred > 0.5).float()
    target = target.float()
    
    # True Positives, False Positives, False Negatives, True Negatives
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    tn = ((1 - pred_bin) * (1 - target)).sum()
    
    # Dice/F1 Score
    smooth = 1.0
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    
    # Sensitivity (Recall)
    se = (tp + smooth) / (tp + fn + smooth)
    
    # Specificity
    sp = (tn + smooth) / (tn + fp + smooth)
    
    # MIoU (Jaccard Index)
    miou = (tp + smooth) / (tp + fp + fn + smooth)
    
    # AUC
    try:
        auc = roc_auc_score(target.cpu().numpy().flatten(), pred.cpu().numpy().flatten())
    except:
        auc = 0.0
    
    # HD95
    def hausdorff95(pred, target):
        pred_edges = pred.cpu().numpy().squeeze()
        target_edges = target.cpu().numpy().squeeze()
        if np.sum(pred_edges) == 0 or np.sum(target_edges) == 0:
            return 0.0
        pred_coords = np.argwhere(pred_edges > 0)
        target_coords = np.argwhere(target_edges > 0)
        return max(directed_hausdorff(pred_coords, target_coords)[0],
                  directed_hausdorff(target_coords, pred_coords)[0])
    
    hd95 = hausdorff95(pred_bin, target)
    
    
    return {
        'dice': dice.item(),
        'se': se.item(),
        'sp': sp.item(),
        'miou': miou.item(),
        'auc': auc,
        'hd95': hd95,
    }

for epoch in range(100):
    model.train()
    metrics = {
        'loss': 0.0,
        'dice': 0.0,
        'se': 0.0,
        'sp': 0.0,
        'miou': 0.0,
        'auc': 0.0,
        'hd95': 0.0,
    }
    
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs, features = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks, features)
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        batch_metrics = calculate_metrics(outputs, masks)
        
        # Accumulate metrics
        metrics['loss'] += loss.item()
        for k in batch_metrics:
            metrics[k] += batch_metrics[k]
    
    # Calculate epoch averages
    num_batches = len(train_loader)
    for k in metrics:
        metrics[k] /= num_batches
    
    # Print metrics
    print(f"\nEpoch {epoch+1:03d} Results:")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Dice/F1: {metrics['dice']:.4f}")
    print(f"Sensitivity (SE): {metrics['se']:.4f}")
    print(f"Specificity (SP): {metrics['sp']:.4f}")
    print(f"HD95: {metrics['hd95']:.4f} px")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"MIoU: {metrics['miou']:.4f}")
    print(f"Connectivity: {metrics['connectivity']:.4f}")
