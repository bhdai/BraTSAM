import torch
import argparse
import json
from torch.utils.data import DataLoader
from transformers import SamProcessor
from monai.metrics.meandice import DiceMetric
from monai.metrics.meaniou import MeanIoU
from tqdm import tqdm

from model import SamFineTuner
from dataset import BrainTumorDataset


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    processor = SamProcessor.from_pretrained(args.model_id)

    model = SamFineTuner(
        model_id=args.model_id,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    print(f"Loading model weights from: {args.checkpoint_path}")
    model.load_state_dict(
        torch.load(args.checkpoint_path, map_location=device), strict=False
    )
    model.to(device)
    model.eval()

    # load metadata and create data splits
    with open(args.test_metadata_path, "r") as f:
        test_metadata = json.load(f)

    test_files = list(test_metadata.keys())

    print(f"Test set size: {len(test_files)} samples")

    test_dataset = BrainTumorDataset(
        image_dir=args.test_image_dir,
        mask_dir=args.test_mask_dir,
        metadata=test_metadata,
        filenames=test_files,
        no_prompt_mode=args.no_prompt_mode,
        processor=processor,
        perturbation_level=args.perturbation_level,  # perturbation is ignored in no-prompt mode
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")

    print("\n--- Starting Evaluation ---")

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
            dice_metric(predicted_binary, ground_truth_masks)
            iou_metric(predicted_binary, ground_truth_masks)

    # Calculate final averaged metrics
    final_dice = dice_metric.aggregate().item()
    final_iou = iou_metric.aggregate().item()

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
        "--test_image_dir",
        type=str,
        default="./data/images_test",
        help="Directory with test images.",
    )
    parser.add_argument(
        "--test_mask_dir",
        type=str,
        default="./data/masks_test",
        help="Directory with test masks.",
    )
    parser.add_argument(
        "--test_metadata_path",
        type=str,
        default="./data/metadata_test.json",
        help="Path to the test metadata JSON file.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA for loading a parameter-efficient fine-tuned model.",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=8, help="The rank 'r' for LoRA matrices."
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="The alpha scaling parameter for LoRA.",
    )
    parser.add_argument(
        "--perturbation_level",
        type=int,
        default=0,
        help="Max pixel perturbation for bbox augmentation during testing. Set to 0 for no perturbation.",
    )
    parser.add_argument(
        "--no_prompt_mode",
        action="store_true",
        help="Enable fully automatic (prompt-less) training and evaluation.",
    )
    args = parser.parse_args()
    main(args)
