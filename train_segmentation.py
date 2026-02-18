"""
Advanced Segmentation Training Script
Optimized for maximum IoU on imbalanced terrain classes
WITHOUT using test data for training (validation only for monitoring)
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
import cv2
import os
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import warnings
import albumentations as A
import math
warnings.filterwarnings('ignore')

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img.cpu())
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def ensure_divisible_by_14(x):
    """Ensure dimension is divisible by 14 for DINOv2 patch embedding"""
    return (x // 14) * 14


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)

# Class names for reporting
class_names = {
    0: "Background",
    1: "Trees",
    2: "Lush Bushes", 
    3: "Dry Grass",
    4: "Dry Bushes",
    5: "Ground Clutter",
    6: "Logs",
    7: "Rocks",
    8: "Landscape",
    9: "Sky"
}


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Advanced IoU-Oriented Loss Functions
# ============================================================================

class LovaszSoftmaxLoss(nn.Module):
    """Lovasz-Softmax loss for direct IoU optimization"""
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        return lovasz_softmax(pred, target, ignore=self.ignore_index)


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss - combination of Focal and Tversky"""
    def __init__(self, alpha=0.3, beta=0.7, gamma=4/3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        # Flatten
        pred = pred.contiguous().view(-1, pred.shape[1])
        target_one_hot = target_one_hot.contiguous().view(-1, target_one_hot.shape[1])
        
        # True Positives, False Positives, False Negatives
        tp = (pred * target_one_hot).sum(dim=0)
        fp = ((1 - target_one_hot) * pred).sum(dim=0)
        fn = (target_one_hot * (1 - pred)).sum(dim=0)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class CombinedLoss(nn.Module):
    """Combination of multiple losses with learnable weights"""
    def __init__(self, class_weights, device):
        super().__init__()
        self.class_weights = class_weights
        self.device = device
        
        # Initialize losses
        self.focal_tversky = FocalTverskyLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=class_weights)
        
        # Learnable loss weights
        self.loss_weights = nn.Parameter(torch.ones(3, device=device))
        
    def forward(self, pred, target):
        focal_tversky_loss = self.focal_tversky(pred, target)
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        
        # Softmax normalized weights
        weights = torch.softmax(self.loss_weights, dim=0)
        
        total_loss = (weights[0] * focal_tversky_loss + 
                      weights[1] * dice_loss + 
                      weights[2] * focal_loss)
        
        return total_loss


# ============================================================================
# Lovasz-Softmax helper function (simplified to avoid errors)
# ============================================================================

def lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors"""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """Simplified Lovasz-Softmax for 2D multi-class segmentation"""
    if per_image:
        loss = 0
        for prob, lab in zip(probas, labels):
            loss += lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
        return loss / len(probas)
    else:
        return lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)


def lovasz_softmax_flat(probas, labels, classes='present'):
    """Lovasz-Softmax for flat predictions"""
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if (classes == 'present' and fg.sum() == 0):
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    
    if not losses:
        return torch.tensor(0.0, device=probas.device)
    return sum(losses) / len(losses)


def flatten_probas(probas, labels, ignore=None):
    """Flatten predictions and labels for Lovasz loss"""
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


# ============================================================================
# Dataset with proper split and class balancing
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_training=True, class_weights=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.is_training = is_training
        self.class_weights = class_weights
        self.data_ids = sorted(os.listdir(self.image_dir))
        
        # Compute sample weights for oversampling if training
        if is_training and class_weights is not None:
            self.sample_weights = self._compute_sample_weights()

    def __len__(self):
        return len(self.data_ids)

    def _compute_sample_weights(self):
        """Compute sample weights for weighted sampling to handle class imbalance"""
        sample_weights = []
        for idx in tqdm(range(len(self)), desc="Computing sample weights"):
            _, mask, _ = self.__getitem__(idx, return_mask_only=True)
            mask = mask.squeeze()
            
            # Weight based on rarest class present
            unique_classes = torch.unique(mask)
            weight = max([self.class_weights[c.item()] for c in unique_classes])
            sample_weights.append(weight.item())
        
        return torch.tensor(sample_weights)

    def __getitem__(self, idx, return_mask_only=False):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        # Load image with OpenCV for Albumentations
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load and convert mask
        mask = Image.open(mask_path)
        mask = convert_mask(mask)
        mask = np.array(mask)

        if return_mask_only:
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            return None, mask_tensor, data_id

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0)

        return image, mask, data_id


def compute_class_weights(dataset, num_classes, device):
    """Compute class weights based on inverse frequency"""
    print("Computing class weights from training dataset...")
    class_counts = torch.zeros(num_classes)
    
    for _, mask, _ in tqdm(dataset, desc="Analyzing class distribution"):
        mask = mask.squeeze()
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum().item()
    
    # Handle missing classes
    class_counts = torch.clamp(class_counts, min=1)
    
    # Inverse frequency weighting with smoothing
    total_pixels = class_counts.sum()
    class_weights = total_pixels / class_counts
    class_weights = class_weights / class_weights.min()  # Normalize
    
    # Print class distribution
    print("\nClass distribution:")
    for i in range(num_classes):
        percentage = (class_counts[i] / total_pixels * 100).item()
        print(f"  {class_names[i]}: {percentage:.2f}%")
    
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    return class_weights.to(device)


# ============================================================================
# Simplified but Effective Segmentation Head
# ============================================================================

class SimpleSegmentationHead(nn.Module):
    """Simplified but effective segmentation head with dynamic dimension support"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Conv2d(256, out_channels, 1)
        
    def forward(self, x, img_h, img_w):
        B, N, C = x.shape
        
        # Dynamically calculate H and W based on image dimensions and DINOv2 patch size (14)
        H = img_h // 14
        W = img_w // 14
        
        # Reshape to spatial format
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        
        return x


# ============================================================================
# EMA (Exponential Moving Average) for model weights
# ============================================================================

class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# Warmup Cosine Learning Rate Scheduler
# ============================================================================

class WarmupCosineLR:
    """Warmup cosine learning rate scheduler"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * lr_scale
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================================
# Metrics with per-class IoU tracking
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    class_ious = {}
    
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou = float('nan')
        else:
            iou = (intersection / union).cpu().numpy()
        
        iou_per_class.append(iou)
        class_ious[class_id] = iou

    return np.nanmean(iou_per_class), class_ious


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)
        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


def multi_scale_inference(model, backbone, imgs, device):
    """Perform multi-scale inference with proper dimension handling"""
    scales = [0.75, 1.0, 1.25]
    logits_list = []
    original_h, original_w = imgs.shape[2:]
    
    for scale in scales:
        # Calculate scaled dimensions and ensure they're at least 14x14
        scaled_h = max(14, ensure_divisible_by_14(int(original_h * scale)))
        scaled_w = max(14, ensure_divisible_by_14(int(original_w * scale)))
        
        # Resize to valid dimensions
        scaled_imgs = F.interpolate(imgs, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
        
        # Forward pass (passing the dynamic height and width of the scaled images)
        output = backbone.forward_features(scaled_imgs)["x_norm_patchtokens"]
        logits = model(output, scaled_h, scaled_w)
        
        # Resize back to original size
        logits = F.interpolate(logits, size=(original_h, original_w), mode="bilinear", align_corners=False)
        logits_list.append(logits)
        
        # Flipped version for TTA
        flipped_imgs = torch.flip(scaled_imgs, dims=[3])
        output = backbone.forward_features(flipped_imgs)["x_norm_patchtokens"]
        logits = model(output, scaled_h, scaled_w)
        logits = F.interpolate(logits, size=(original_h, original_w), mode="bilinear", align_corners=False)
        logits = torch.flip(logits, dims=[3])
        logits_list.append(logits)
    
    # Average predictions
    return torch.mean(torch.stack(logits_list), dim=0)


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, show_progress=True, multi_scale=False):
    """Evaluate all metrics on a dataset with optional multi-scale inference."""
    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    all_class_ious = {i: [] for i in range(num_classes)}

    model.eval()
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") if show_progress else data_loader
    
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            if multi_scale:
                outputs = multi_scale_inference(model, backbone, imgs, device)
            else:
                output = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(output, imgs.shape[2], imgs.shape[3])
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels = labels.squeeze(dim=1).long()

            iou, class_ious = compute_iou(outputs, labels, num_classes=num_classes)
            dice = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            
            for c in range(num_classes):
                if not np.isnan(class_ious[c]):
                    all_class_ious[c].append(class_ious[c])

    model.train()
    
    # Calculate per-class IoU
    per_class_iou = {}
    for c in range(num_classes):
        if all_class_ious[c]:
            per_class_iou[c] = np.mean(all_class_ious[c])
        else:
            per_class_iou[c] = float('nan')
    
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies), per_class_iou


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train', linewidth=2)
    plt.plot(history['val_loss'], label='val', linewidth=2)
    plt.title('Loss Curves', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train', linewidth=2)
    plt.plot(history['val_pixel_acc'], label='val', linewidth=2)
    plt.title('Pixel Accuracy', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()

    # Plot 2: IoU curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_iou'], label='Train IoU', linewidth=2)
    plt.title('Train IoU vs Epoch', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU', linewidth=2)
    plt.title('Validation IoU vs Epoch', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'), dpi=150)
    plt.close()

    # Plot 3: Combined metrics
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train', linewidth=2)
    plt.plot(history['val_loss'], label='val', linewidth=2)
    plt.title('Loss vs Epoch', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'], label='train', linewidth=2)
    plt.plot(history['val_iou'], label='val', linewidth=2)
    plt.title('IoU vs Epoch', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='train', linewidth=2)
    plt.plot(history['val_dice'], label='val', linewidth=2)
    plt.title('Dice Score vs Epoch', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(history['train_pixel_acc'], label='train', linewidth=2)
    plt.plot(history['val_pixel_acc'], label='val', linewidth=2)
    plt.title('Pixel Accuracy vs Epoch', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'), dpi=150)
    plt.close()


def save_history_to_file(history, per_class_ious, output_dir):
    """Save training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("FINAL METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n\n")

        f.write("BEST RESULTS:\n")
        f.write("-" * 40 + "\n")
        best_val_iou_epoch = np.argmax(history['val_iou'])
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {best_val_iou_epoch + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n\n")

        f.write("PER-CLASS IOUs (Best Model):\n")
        f.write("-" * 40 + "\n")
        for class_id, iou in per_class_ious.items():
            if not np.isnan(iou):
                f.write(f"  {class_names[class_id]}: {iou:.4f}\n")
            else:
                f.write(f"  {class_names[class_id]}: N/A (no samples)\n")
        f.write("\n")

        f.write("PER-EPOCH HISTORY:\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_iou'][i],
                history['val_iou'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_pixel_acc'][i],
                history['val_pixel_acc'][i]
            ))

    print(f"Saved evaluation metrics to {filepath}")


def visualize_predictions(model, backbone, dataset, device, num_samples=5, output_dir='predictions'):
    """Visualize model predictions on sample images."""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask, data_id = dataset[idx]
            original_image = image.clone()
            
            # Add batch dimension
            image = image.unsqueeze(0).to(device)
            
            # Get prediction with multi-scale inference
            outputs = multi_scale_inference(model, backbone, image, device)
            pred = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
            
            # Convert tensors to numpy for visualization
            image_np = original_image.cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(np.transpose(image_np, (1, 2, 0)))
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(mask_np, cmap='tab20', vmin=0, vmax=n_classes-1)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(pred, cmap='tab20', vmin=0, vmax=n_classes-1)
            axes[2].set_title('Prediction (Multi-scale)')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'prediction_{i}_{data_id}.png'), dpi=150)
            plt.close()
    
    model.train()
    print(f"Saved {num_samples} prediction visualizations to {output_dir}")


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Hyperparameters - ensure dimensions are divisible by 14
    base_h, base_w = 540, 960
    h = ensure_divisible_by_14(base_h // 2)
    w = ensure_divisible_by_14(base_w // 2)
    
    batch_size = 2
    lr = 1e-4
    n_epochs = 50
    val_split = 0.15
    warmup_epochs = 5
    grad_accum_steps = 4
    ema_decay = 0.999

    print(f"Training dimensions: {h}x{w} (divisible by 14)")

    # Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    pred_dir = os.path.join(script_dir, 'predictions')
    checkpoint_dir = os.path.join(script_dir, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training Augmentations
    train_transform = A.Compose([
        A.Resize(height=h, width=w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Validation Transform
    val_transform = A.Compose([
        A.Resize(height=h, width=w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Dataset path
    data_dir = r"C:\Users\parim\OneDrive\Desktop\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train"

    # Load full dataset
    print("Loading dataset...")
    full_dataset = MaskDataset(data_dir=data_dir, transform=train_transform, is_training=True)
    
    # Compute class weights first
    class_weights = compute_class_weights(full_dataset, n_classes, device)
    
    # Recreate dataset with class weights for sampling
    full_dataset = MaskDataset(data_dir=data_dir, transform=train_transform, 
                               is_training=True, class_weights=class_weights)
    
    # Split into train and validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Apply different transforms to validation set
    val_dataset.dataset.transform = val_transform
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create weighted sampler
    train_weights = [full_dataset.sample_weights[i] for i in train_dataset.indices]
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)

    # Load DINOv2 backbone
    print("\nLoading DINOv2 backbone...")
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
    backbone_model.to(device)
    
    # Freeze backbone
    for param in backbone_model.parameters():
        param.requires_grad = False
    print("Backbone loaded and frozen!")

    # Get embedding dimension
    sample_imgs, _, _ = next(iter(train_loader))
    sample_imgs = sample_imgs.to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(sample_imgs)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")
    print(f"Patch tokens shape: {output.shape}")

    # Create segmentation head - dynamic version
    classifier = SimpleSegmentationHead(
        in_channels=n_embedding,
        out_channels=n_classes
    )
    classifier = classifier.to(device)

    # Loss with learnable weights
    criterion = CombinedLoss(class_weights=class_weights, device=device)
    
    # Optimizer
    optimizer = optim.AdamW([
        {'params': classifier.parameters(), 'lr': lr},
        {'params': criterion.loss_weights, 'lr': lr * 0.1}
    ], weight_decay=1e-4)
    
    # Warmup cosine scheduler
    scheduler = WarmupCosineLR(optimizer, warmup_epochs, n_epochs, min_lr=1e-6)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # EMA for stability
    ema = EMA(classifier, decay=ema_decay)

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'train_pixel_acc': [], 'val_pixel_acc': []
    }

    # Best model tracking
    best_val_iou = 0.0
    best_model_path = os.path.join(script_dir, "best_segmentation_head.pth")
    best_per_class_ious = {}

    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    optimizer.zero_grad()
    for epoch in range(n_epochs):
        # Training phase
        classifier.train()
        train_losses = []
        accumulated_loss = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", 
                          leave=False, unit="batch")
        
        for batch_idx, (imgs, labels, _) in enumerate(train_pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast():
                with torch.no_grad():
                    output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

                # Pass dynamic height and width from the input image
                logits = classifier(output, imgs.shape[2], imgs.shape[3])
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                labels = labels.squeeze(dim=1).long()
                loss = criterion(outputs, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            accumulated_loss += loss.item() * grad_accum_steps

            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update()
                train_pbar.set_postfix(loss=f"{accumulated_loss:.4f}")
                accumulated_loss = 0

            train_losses.append(loss.item() * grad_accum_steps)

        # Validation phase with EMA
        ema.apply_shadow()
        classifier.eval()
        val_losses = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", 
                            leave=False, unit="batch")
            for imgs, labels, _ in val_pbar:
                imgs, labels = imgs.to(device), labels.to(device)

                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                # Pass dynamic height and width from the input image
                logits = classifier(output, imgs.shape[2], imgs.shape[3])
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

                labels = labels.squeeze(dim=1).long()
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

        ema.restore()

        # Calculate metrics
        train_iou, train_dice, train_pixel_acc, _ = evaluate_metrics(
            classifier, backbone_model, train_loader, device, num_classes=n_classes
        )
        val_iou, val_dice, val_pixel_acc, per_class_ious = evaluate_metrics(
            classifier, backbone_model, val_loader, device, num_classes=n_classes, multi_scale=True
        )

        # Store history
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['val_pixel_acc'].append(val_pixel_acc)

        # Update learning rate
        scheduler.step(epoch)

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_per_class_ious = per_class_ious
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'ema_state_dict': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'per_class_ious': per_class_ious,
                'loss_weights': criterion.loss_weights.detach().cpu().numpy()
            }, best_model_path)
            print(f"\n✨ New best model saved! Val IoU: {val_iou:.4f}")

        # Print epoch summary
        current_lr = optimizer.param_groups[0]['lr']
        loss_weights = torch.softmax(criterion.loss_weights, dim=0).detach().cpu().numpy()
        
        print(f"\n📊 Epoch {epoch+1}/{n_epochs} Summary:")
        print(f"  ├─ Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(f"  ├─ Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}")
        print(f"  ├─ Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")
        print(f"  ├─ Train Acc: {train_pixel_acc:.4f} | Val Acc: {val_pixel_acc:.4f}")
        print(f"  ├─ LR: {current_lr:.2e}")
        print(f"  └─ Loss Weights: FocTv={loss_weights[0]:.2f}, Dice={loss_weights[1]:.2f}, Foc={loss_weights[2]:.2f}")
        print("-" * 60)

    # Save plots and final model
    print("\n💾 Saving training curves...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, best_per_class_ious, output_dir)

    final_model_path = os.path.join(script_dir, "segmentation_head.pth")
    torch.save(classifier.state_dict(), final_model_path)
    print(f"✅ Saved final model to '{final_model_path}'")
    print(f"✅ Saved best model to '{best_model_path}'")

    # Final evaluation
    print("\n📸 Generating prediction visualizations...")
    checkpoint = torch.load(best_model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    visualize_predictions(classifier, backbone_model, val_dataset, device, num_samples=10, output_dir=pred_dir)

    print("\n" + "="*60)
    print("🏆 FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"  Best Val IoU:      {best_val_iou:.4f}")
    print(f"  Final Val Loss:    {history['val_loss'][-1]:.4f}")
    print(f"  Final Val IoU:     {history['val_iou'][-1]:.4f}")
    
    print("\n📈 Per-class IoU (Best Model):")
    for class_id, iou in best_per_class_ious.items():
        if not np.isnan(iou):
            print(f"  {class_names[class_id]:<15}: {iou:.4f}")

    print("\n🎉 Training complete! Best validation IoU: {:.4f}".format(best_val_iou))


if __name__ == "__main__":
    main()