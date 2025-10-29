import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from monai.metrics.meandice import DiceMetric
from monai.metrics.meaniou import MeanIoU


def evaluate_predictions(pred_dir, gt_dir):
    """
    Calculates Dice and IoU scores by comparing predicted masks against ground truth masks.

    Args:
        pred_dir (str): Directory containing the predicted segmentation masks (from nnU-Net).
        gt_dir (str): Directory containing the ground truth segmentation masks.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize MONAI metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")

    # Get list of ground truth files to iterate over
    gt_filenames = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])

    if not gt_filenames:
        print(f"Error: No ground truth masks found in {gt_dir}")
        return

    print(f"\n--- Starting Evaluation on {len(gt_filenames)} samples ---")
    progress_bar = tqdm(gt_filenames, desc="Evaluating nnU-Net Predictions")

    for gt_filename in progress_bar:
        gt_path = os.path.join(gt_dir, gt_filename)

        # nnU-Net output matches the original mask filename convention
        # e.g., patientX_mask_085.png -> patientX_slice_085.png in SAM's test set
        # nnU-Net prediction name: patientX_slice_085.png
        # map gt mask name to prediction name
        pred_filename = gt_filename.replace("_mask_", "_slice_")
        pred_path = os.path.join(pred_dir, pred_filename)

        if not os.path.exists(pred_path):
            tqdm.write(
                f"Warning: Prediction for {gt_filename} not found at {pred_path}. Skipping."
            )
            continue

        # Load images
        gt_mask_img = Image.open(gt_path).convert("L")
        pred_mask_img = Image.open(pred_path).convert("L")

        gt_mask = torch.from_numpy(np.array(gt_mask_img)).float().to(device)
        pred_mask = torch.from_numpy(np.array(pred_mask_img)).float().to(device)

        # e masks (ensure values are 0 or 1
        gt_mask = (gt_mask > 0).float()
        pred_mask = (pred_mask > 0).float()

        # MONAI metrics expect batch and channel dimensions: [B, C, H, W]
        # Here B=1, C=1
        gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)
        pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)

        # Accumulate metrics
        dice_metric(y_pred=pred_mask, y=gt_mask)
        iou_metric(y_pred=pred_mask, y=gt_mask)

    # aggregate the final metrics
    final_dice = dice_metric.aggregate().item()
    final_iou = iou_metric.aggregate().item()

    # reset metrics for future use if needed
    dice_metric.reset()
    iou_metric.reset()

    print("\n--- Evaluation Complete ---")
    print(f"Final nnU-Net Test Dice Score: {final_dice:.4f}")
    print(f"Final nnU-Net Test IoU Score:  {final_iou:.4f}")
    print(f"Evaluated on {len(gt_filenames)} test samples.")


if __name__ == "__main__":
    pred_dir = "./nnunet/nnunet_test_predictions"

    gt_mask_dir = "./data/masks_test"

    evaluate_predictions(pred_dir, gt_mask_dir)
