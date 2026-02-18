"""
Advanced Segmentation Validation Script
Matches the training script architecture and features
Evaluates a trained segmentation head on validation data and saves predictions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm
import warnings
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

# Class names for visualization
class_names_list = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

class_names_dict = {
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

# Color palette for visualization (10 distinct colors)
color_palette = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Dataset (matches training script)
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        # Load image with OpenCV (matches training)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load and convert mask
        mask = Image.open(mask_path)
        mask = convert_mask(mask)
        mask = np.array(mask)

        if self.transform:
            # Apply the same transform type as training
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0)

        return image, mask, data_id


# ============================================================================
# Model: SimpleSegmentationHead (must match training)
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
        
        # Calculate H and W based on image dimensions and DINOv2 patch size (14)
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
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    """Compute IoU for each class and return mean IoU."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


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

    return np.mean(dice_per_class), dice_per_class


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


def multi_scale_inference(model, backbone, imgs, device):
    """Perform multi-scale inference with proper dimension handling (matches training)"""
    scales = [0.75, 1.0, 1.25]
    logits_list = []
    original_h, original_w = imgs.shape[2:]
    
    for scale in scales:
        # Calculate scaled dimensions and ensure they're at least 14x14
        scaled_h = max(14, ensure_divisible_by_14(int(original_h * scale)))
        scaled_w = max(14, ensure_divisible_by_14(int(original_w * scale)))
        
        # Resize to valid dimensions
        scaled_imgs = F.interpolate(imgs, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
        
        # Forward pass
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


# ============================================================================
# Visualization Functions
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    """Save a side-by-side comparison of input, ground truth, and prediction."""
    # Denormalize image
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # Convert masks to color
    gt_color = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(gt_color)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir):
    """Save metrics summary to a text file and create bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    # Save text summary
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write(f"Mean Dice:         {results['mean_dice']:.4f}\n")
        f.write(f"Pixel Accuracy:    {results['pixel_acc']:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for i, name in enumerate(class_names_list):
            iou = results['class_iou'][i] if i < len(results['class_iou']) else float('nan')
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {iou_str}\n")
        
        f.write("\nPer-Class Dice:\n")
        f.write("-" * 40 + "\n")
        for i, name in enumerate(class_names_list):
            dice = results['class_dice'][i] if i < len(results['class_dice']) else float('nan')
            dice_str = f"{dice:.4f}" if not np.isnan(dice) else "N/A"
            f.write(f"  {name:<20}: {dice_str}\n")

    print(f"\nSaved evaluation metrics to {filepath}")

    # Create bar chart for per-class IoU
    fig, ax = plt.subplots(figsize=(12, 6))

    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    bars = ax.bar(range(n_classes), valid_iou, color=[color_palette[i] / 255 for i in range(n_classes)],
                  edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, iou in zip(bars, valid_iou):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{iou:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names_list, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title(f'Per-Class IoU (Mean: {results["mean_iou"]:.4f})', fontsize=14)
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', linewidth=2, label=f'Mean IoU: {results["mean_iou"]:.4f}')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create bar chart for per-class Dice
    fig, ax = plt.subplots(figsize=(12, 6))

    valid_dice = [dice if not np.isnan(dice) else 0 for dice in results['class_dice']]
    bars = ax.bar(range(n_classes), valid_dice, color=[color_palette[i] / 255 for i in range(n_classes)],
                  edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, dice in zip(bars, valid_dice):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{dice:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names_list, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title(f'Per-Class Dice Score (Mean: {results["mean_dice"]:.4f})', fontsize=14)
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_dice'], color='red', linestyle='--', linewidth=2, label=f'Mean Dice: {results["mean_dice"]:.4f}')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_dice.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved per-class metrics charts to '{output_dir}/per_class_iou.png' and '{output_dir}/per_class_dice.png'")


# ============================================================================
# Main Validation Function
# ============================================================================

def main():
    # Get script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='Advanced segmentation prediction/inference script')
    parser.add_argument('--model_path', type=str, default=os.path.join(script_dir, 'best_segmentation_head.pth'),
                        help='Path to trained model weights (use best model)')
    parser.add_argument('--data_dir', type=str, default=os.path.join(script_dir, '..', 'Offroad_Segmentation_testImages'),
                        help='Path to validation/test dataset')
    parser.add_argument('--output_dir', type=str, default='./advanced_predictions',
                        help='Directory to save prediction visualizations')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for validation')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of comparison visualizations to save')
    parser.add_argument('--multi_scale', action='store_true', default=True,
                        help='Use multi-scale inference')
    parser.add_argument('--save_raw', action='store_true', default=True,
                        help='Save raw prediction masks')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Validation transform (matches training validation transform)
    from albumentations import Compose, Resize, Normalize
    from albumentations.pytorch import ToTensorV2
    
    val_transform = Compose([
        Resize(height=270, width=480),  # h=270, w=480 from training
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Create dataset
    print(f"Loading dataset from {args.data_dir}...")
    valset = MaskDataset(data_dir=args.data_dir, transform=val_transform)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Loaded {len(valset)} samples")

    # Load DINOv2 backbone (same as training)
    print("Loading DINOv2 backbone...")
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    # Get embedding dimension
    sample_img, _, _ = valset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(sample_img)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")

    # Load classifier (using SimpleSegmentationHead from training)
    print(f"Loading model from {args.model_path}...")
    classifier = SimpleSegmentationHead(
        in_channels=n_embedding,
        out_channels=n_classes
    )
    
    # Load weights (handle both full checkpoint and state dict)
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
        if 'loss_weights' in checkpoint:
            print(f"Loss weights from training: {checkpoint['loss_weights']}")
        if 'val_iou' in checkpoint:
            print(f"Validation IoU from training: {checkpoint['val_iou']:.4f}")
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device)
    classifier.eval()
    print("Model loaded successfully!")

    # Create subdirectories for outputs
    masks_dir = os.path.join(args.output_dir, 'masks_raw')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    # Run evaluation and save predictions
    print(f"\nRunning evaluation and saving predictions for all {len(valset)} images...")
    if args.multi_scale:
        print("Using multi-scale inference with TTA")

    iou_scores = []
    dice_scores = []
    pixel_accuracies = []
    all_class_iou = []
    all_class_dice = []
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Processing", unit="batch")
        for batch_idx, (imgs, labels, data_ids) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward pass with optional multi-scale
            if args.multi_scale:
                outputs = multi_scale_inference(classifier, backbone_model, imgs, device)
            else:
                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(output, imgs.shape[2], imgs.shape[3])
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            labels_squeezed = labels.squeeze(dim=1).long()
            predicted_masks = torch.argmax(outputs, dim=1)

            # Calculate metrics
            iou, class_iou = compute_iou(outputs, labels_squeezed, num_classes=n_classes)
            dice, class_dice = compute_dice(outputs, labels_squeezed, num_classes=n_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels_squeezed)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            all_class_iou.append(class_iou)
            all_class_dice.append(class_dice)

            # Save predictions
            for i in range(imgs.shape[0]):
                data_id = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                if args.save_raw:
                    # Save raw prediction mask (class IDs 0-9)
                    pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)
                    pred_img = Image.fromarray(pred_mask)
                    pred_img.save(os.path.join(masks_dir, f'{base_name}_pred.png'))

                # Save colored prediction mask (RGB visualization)
                pred_mask_np = predicted_masks[i].cpu().numpy().astype(np.uint8)
                pred_color = mask_to_color(pred_mask_np)
                cv2.imwrite(os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                            cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Save comparison visualization for first N samples
                if sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i], labels_squeezed[i], predicted_masks[i],
                        os.path.join(comparisons_dir, f'sample_{sample_count:03d}_{base_name}_comparison.png'),
                        data_id
                    )

                sample_count += 1

            # Update progress bar with metrics
            pbar.set_postfix(iou=f"{iou:.3f}", dice=f"{dice:.3f}")

    # Aggregate results
    mean_iou = np.nanmean(iou_scores)
    mean_dice = np.nanmean(dice_scores)
    mean_pixel_acc = np.mean(pixel_accuracies)

    # Average per-class metrics
    avg_class_iou = np.nanmean(all_class_iou, axis=0)
    avg_class_dice = np.nanmean(all_class_dice, axis=0)

    results = {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'pixel_acc': mean_pixel_acc,
        'class_iou': avg_class_iou,
        'class_dice': avg_class_dice
    }

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean IoU:          {mean_iou:.4f}")
    print(f"Mean Dice:         {mean_dice:.4f}")
    print(f"Pixel Accuracy:    {mean_pixel_acc:.4f}")
    print("=" * 50)
    
    print("\nPer-Class IoU:")
    for i, name in enumerate(class_names_list):
        iou_val = avg_class_iou[i] if i < len(avg_class_iou) else float('nan')
        print(f"  {name:<20}: {iou_val:.4f}")

    # Save all results
    save_metrics_summary(results, args.output_dir)

    print(f"\nPrediction complete! Processed {len(valset)} images.")
    print(f"\nOutputs saved to {args.output_dir}/")
    if args.save_raw:
        print(f"  - masks_raw/        : Raw prediction masks (class IDs 0-9)")
    print(f"  - masks_color/      : Colored prediction masks (RGB)")
    print(f"  - comparisons/      : Side-by-side comparison images ({min(sample_count, args.num_samples)} samples)")
    print(f"  - evaluation_metrics.txt")
    print(f"  - per_class_iou.png")
    print(f"  - per_class_dice.png")


if __name__ == "__main__":
    main()