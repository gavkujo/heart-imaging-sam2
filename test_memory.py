import os
import glob
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
from medpy.metric.binary import dc, hd95
from sam2_video_predictor import SAM2VideoPredictor

# Paths
IMG_PATH = "./acdc_data/img"
GT_PATH = "./acdc_data/gt"

# Model init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAM2VideoPredictor()
model.eval().to(device)

# Get all test images (just a few for sanity)
img_files = sorted(glob.glob(os.path.join(IMG_PATH, "*.nii.gz")))[:3]  # grab 3 frames

def load_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))  # returns [slices, H, W]

def preprocess(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def convert_to_rgb(img):
    """Convert single-channel grayscale image to RGB (3 channels)."""
    img_rgb = np.stack([img]*3, axis=-1) * 255  # [H, W, 3] with values [0, 255]
    img_rgb = img_rgb.astype(np.uint8)
    return img_rgb

def compute_metrics(pred, gt):
    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin = gt.astype(np.uint8)

    dice = dc(pred_bin, gt_bin)
    haus = hd95(pred_bin, gt_bin)
    return dice, haus

all_dice, all_hd = [], []

print(f"Testing on {len(img_files)} files...")

for img_path in tqdm(img_files):
    filename = os.path.basename(img_path)
    gt_path = os.path.join(GT_PATH, filename.replace(".nii.gz", "_gt.nii.gz"))

    # Load and prep
    img = load_nii(img_path)[0]  # [1, H, W] since single-slice
    gt = load_nii(gt_path)[0]

    img = preprocess(img)
    img_rgb = convert_to_rgb(img)  # Convert grayscale to RGB

    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]

    # Predict
    with torch.no_grad():
        pred = model(input_tensor)  # [1, 1, H, W]
        pred_np = pred.squeeze().cpu().numpy()

    # Metrics
    dice, haus = compute_metrics(pred_np, gt)
    all_dice.append(dice)
    all_hd.append(haus)

    print(f"{filename}: Dice = {dice:.4f}, HD95 = {haus:.2f}")

# Summary
print("\n=== Summary ===")
print(f"Mean Dice: {np.mean(all_dice):.4f}")
print(f"Mean HD95: {np.mean(all_hd):.2f}")
