import torch
import argparse
import json
from torch.utils.data import DataLoader
from transformers import SamProcessor
from monai.metrics.meandice import DiceMetric
from monai.metrics.meaniou import MeanIoU
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import SamFineTuner
from dataset import BrainTumorDataset


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    processor = SamProcessor.from_pretrained(args.model_id)

    model = SamFineTuner(model_id=args.model_id)

    print(f"Loading model weights from: {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # load metadata and create data splits
    with open(args.metadata_path, "r") as f:
        metadata = json.load(f)

    all_image_files = list(metadata.keys())

    train_files, temp_files = train_test_split(
        all_image_files, test_size=0.2, random_state=42
    )
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    print(f"Test set size: {len(test_files)} samples")

    test_dataset = BrainTumorDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        metadata=metadata,
        filenames=test_files,
        processor=processor,
        perturbation_level=0,  # no bbox noise during testing
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    iou_metric = MeanIoU(include_background=False, reduction="mean_batch")

    print("\n--- Starting Evaluation ---")

    all_dice_scores = []
    all_iou_scores = []

    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Evaluating")

        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            ground_truth_masks = batch["labels"].to(device)

            predicted_masks = model(pixel_values=pixel_values, input_boxes=input_boxes)

            # convert logits to binary predictions
            predicted_binary = (torch.sigmoid(predicted_masks) > 0.5).float()

            # add channel dimension for monai metrics expects shape (B, C, H, W)
            predicted_binary = predicted_binary.unsqueeze(1)
            ground_truth_masks = ground_truth_masks.unsqueeze(1)

            # Calculate metrics for this batch
            dice_score = dice_metric(predicted_binary, ground_truth_masks)
            iou_score = iou_metric(predicted_binary, ground_truth_masks)

            # Store batch scores
            all_dice_scores.append(dice_score.mean().item())
            all_iou_scores.append(iou_score.mean().item())

            # Update progress bar
            progress_bar.set_postfix(
                dice=f"{dice_score.mean().item():.4f}",
                iou=f"{iou_score.mean().item():.4f}",
            )

    # Calculate final averaged metrics
    final_dice = sum(all_dice_scores) / len(all_dice_scores)
    final_iou = sum(all_iou_scores) / len(all_iou_scores)

    print("\n--- Evaluation Complete ---")
    print(f"Final Test Dice Score: {final_dice:.4f}")
    print(f"Final Test IoU Score:  {final_iou:.4f}")
    print(f"Evaluated on {len(test_files)} test samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test fine-tuned SAM model on held-out test set"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./models/best_model.pth",
        help="Path to the saved model checkpoint.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/sam-vit-base",
        help="The Hugging Face model ID for SAM.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./data/images",
        help="Directory with test images.",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="./data/masks",
        help="Directory with test masks.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="./data/metadata.json",
        help="Path to the metadata JSON file.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation."
    )
    args = parser.parse_args()
    main(args)
