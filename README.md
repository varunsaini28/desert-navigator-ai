<div align="center">
  <h1>🏜️ Desert Navigator AI</h1>
  <h3>Offroad Semantic Segmentation for Autonomous Navigation</h3>

  

  <p align="center">
    <b>Team BinaryBrains</b> 
   
  <a href="https://docs-96pqzsfmb-vs-projects-fe1787c0.vercel.app/"> click here for Interactive Docs</a>
  </p>

 
</div>

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Achievements](#-key-achievements)
- [Class Definitions](#-class-definitions)
- [Project Structure](#-project-structure)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Features](#-features)
- [Interactive Documentation](#-interactive-documentation)
- [Links](#-links)

---

## 🌟 Overview

**Desert Navigator AI** is an advanced semantic segmentation solution for off-road terrain classification, developed for the **Duality AI Hackathon** by Team **BinaryBrains**. The system uses a **DINOv2 backbone** with a custom segmentation head to accurately classify 10 distinct terrain classes in desert environments.

<div align="center">
  <table>
    <tr>
      <td align="center"><b>🎯 Goal</b></td>
      <td align="center"><b>🧠 Architecture</b></td>
      <td align="center"><b>📊 Classes</b></td>
    </tr>
    <tr>
      <td>Off-road autonomous navigation</td>
      <td>DINOv2 + Custom Head</td>
      <td>10 terrain types</td>
    </tr>
  </table>
</div>

---

## 🏆 Key Achievements

<div align="center">
  <table>
    <tr>
      <td align="center"><h1>0.5325</h1><p>Best Validation IoU</p><sub>↑ 15.6% improvement</sub></td>
      <td align="center"><h1>0.4923</h1><p>Test Mean IoU</p><sub>on 1002 unseen images</sub></td>
      <td align="center"><h1>0.9554</h1><p>Sky Segmentation</p><sub>Excellent performance</sub></td>
    </tr>
    <tr>
      <td align="center"><h1>0.5742</h1><p>Dry Grass IoU</p></td>
      <td align="center"><h1>0.5666</h1><p>Trees IoU</p></td>
      <td align="center"><h1>0.7834</h1><p>Best Validation Accuracy</p></td>
    </tr>
  </table>
</div>

---

## 📊 Class Definitions

<div align="center">

| Class ID | Class Name      | Description                     | Color |
|:--------:|-----------------|---------------------------------|:-----:|
| 0        | **Background**  | General ground                  |   ⚫  |
| 1        | **Trees**       | Trees and large vegetation      |   🟢  |
| 2        | **Lush Bushes** | Green, dense bushes             |   🟢  |
| 3        | **Dry Grass**   | Dry/dead grass                  |   🟡  |
| 4        | **Dry Bushes**  | Dry, sparse bushes              |   🟤  |
| 5        | **Ground Clutter** | Small debris, rocks, etc.    |   ⚪  |
| 6        | **Logs**        | Fallen trees/wood               |   🟤  |
| 7        | **Rocks**       | Rock formations                 |   ⚪  |
| 8        | **Landscape**   | Terrain features                |   🟠  |
| 9        | **Sky**         | Sky region                      |   🔵  |

</div>

---

## 📁 Project Structure

📦 OFFROAD_SEGMENTATION_SCRIPTS
├── 📂 checkpoints/ # Model checkpoints during training
├── 📂 ENV_SETUP/ # Environment setup scripts
├── 📂 predictions/ # Generated prediction visualizations
├── 📂 train_stats/ # Training metrics and plots
│
├── 📄 best_segmentation_head.pth # 🏆 Best model weights (Val IoU: 0.5325)
├── 📄 segmentation_head.pth # 📦 Final model weights
├── 📄 train_segmentation.py # 🏋️ Main training script
├── 📄 test_segmentation.py # 🔬 Evaluation and inference script
├── 📄 visualize.py # 🎨 Visualization utilities
└── 📄 README.md # 📖 This file


🏋️ Training

python train_segmentation.py


Training Configuration
<div align="center">
Parameter	Value
Batch Size	2
Learning Rate	1e-4
Epochs	50
Image Size	270×480
Loss Function	CombinedLoss (FocalTversky + Dice + Focal)
Optimizer	AdamW
Weight Decay	1e-4
Scheduler	WarmupCosine (5 epochs)
</div>
Training Outputs
After training, the following charts are saved in train_stats/:

<div align="center"> <table> <tr><td align="center"><b>📈 all_metrics_curves.png</b></td><td align="center"><b>📉 iou_curves.png</b></td></tr> <tr><td align="center"><b>🎲 dice_curves.png</b></td><td align="center"><b>📊 training_curves.png</b></td></tr> </table> </div>


📊 Results
Final Model Performance

TRAINING RESULTS
============================================================
FINAL METRICS:
----------------------------------------
  Final Train Loss:     0.7566
  Final Val Loss:       0.8668
  Final Train IoU:      0.5325
  Final Val IoU:        0.5215
  Best Val IoU:         0.5325 (Epoch 46)
  Best Val Accuracy:    0.7834 (Epoch 46)

TEST RESULTS:
----------------------------------------
  Mean IoU:             0.4923 (1002 images)
  Mean Dice:            0.4373
  Pixel Accuracy:       0.4962


  Per-Class Performance
<div align="center">
📈 Training Validation IoU
Class	IoU	Performance
Sky	0.9554	🟢🟢🟢🟢🟢
Dry Grass	0.5742	🟢🟢🟢🟡
Trees	0.5666	🟢🟢🟢🟡
Lush Bushes	0.5182	🟢🟢🟢
Background	0.4760	🟢🟢
Landscape	0.4697	🟢🟢
Dry Bushes	0.2942	🟡
Rocks	0.2651	🟡
Ground Clutter	0.2471	🟠
Logs	0.1632	🔴
🧪 Test Results IoU
Class	IoU	Performance
Sky	0.9560	🟢🟢🟢🟢🟢
Dry Grass	0.5062	🟢🟢🟢
Landscape	0.4663	🟢🟢
Trees	0.4160	🟢🟢
Dry Bushes	0.2611	🟡
Rocks	0.1694	🟠
Other classes	0.0000*	⚪
<sub>*Note: Some classes were absent in the test set.</sub>

</div>
⚙️ Features
🧠 Advanced Loss Functions
<div align="center">
Loss	Benefit
🔥 Focal Tversky Loss	Handles class imbalance
🎲 Dice Loss	Optimizes overlap
⚡ Focal Loss	Focuses on hard examples
⚖️ Learnable loss weights	Automatically balances contributions
</div>
🚀 Training Optimizations
Class-weighted sampling - Addresses severe imbalance

Mixed precision training - 40% memory reduction

Multi-scale inference - Improves accuracy

Test-time augmentation - Robust predictions

Exponential Moving Average (EMA) - Stable training

🎨 Data Augmentation
python
✓ Random horizontal/vertical flips
✓ Color jittering
✓ Gaussian blur
✓ Gaussian noise
✓ Random brightness/contrast
📱 Interactive Documentation
Explore a web-based, interactive version of this documentation for a better visual experience.


The interactive docs feature:

📑 Sidebar navigation through all 8 pages

📊 Live metrics visualization

✨ Animated background effects

📱 Responsive design for all devices

Docs link : https://docs-96pqzsfmb-vs-projects-fe1787c0.vercel.app/
