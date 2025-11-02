import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import SamProcessor
from monai.losses.dice import DiceCELoss
import os
import argparse
import json
import time
from sklearn.model_selection import train_test_split

from model import SamFineTuner
from dataset import BrainTumorDataset
from engine import train_one_epoch, evaluate, EarlyStopping


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    processor = SamProcessor.from_pretrained(args.model_id)
    model = SamFineTuner(
        model_id=args.model_id,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = DiceCELoss(sigmoid=True)

    # gradscaler for mixed precision training
    scaler = None
    use_amp = args.use_amp and device.type == "cuda"
    if use_amp:
        scaler = torch.amp.GradScaler(device=device.type)
        print("Mixed precision training ENABLED")
    else:
        if args.use_amp and device.type != "cuda":
            print(
                "Warning: mixed precision requested but CUDA not available. Using FP32"
            )
        else:
            print("Mixed precision training DISABLED")

    with open(args.metadata_path, "r") as f:
        metadata = json.load(f)

    all_image_files = list(metadata.keys())

    train_files, val_files = train_test_split(
        all_image_files, test_size=0.1, random_state=42
    )

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
    latest_checkpoint_path = os.path.join(args.output_dir, "latest_model.pth")

    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        verbose=True,
        path=checkpoint_path,
        latest_path=latest_checkpoint_path,
    )

    train_dataset = BrainTumorDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        metadata=metadata,
        filenames=train_files,
        processor=processor,
        perturbation_level=args.perturbation_level,
    )
    val_dataset = BrainTumorDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        metadata=metadata,
        filenames=val_files,
        processor=processor,
        perturbation_level=0,
    )

    print(
        f"Data split: {len(train_dataset)} training, {len(val_dataset)} validation samples."
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")

        epoch_start_time = time.time()
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, loss_fn, device, scaler=scaler
        )
        val_loss = evaluate(model, val_dataloader, loss_fn, device, use_amp=use_amp)
        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch + 1} Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s"
        )

        early_stopper(val_loss, model)

        if early_stopper.early_stop:
            print("Early stoppping triggered")
            break

    print("\n--- Training Complete ---")
    print(f"Best validation loss achieved: {early_stopper.val_loss_min:.4f}")
    print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune SAM for Brain Tumor Segmentation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save model checkpoints.",
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
        help="Directory with training images.",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="./data/masks",
        help="Directory with training masks.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="./data/metadata_train.json",
        help="Path to the metadata JSON file.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--perturbation_level",
        type=int,
        default=10,
        help="Max pixel perturbation for bbox augmentation.",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable automatic mixed precision training (FP16/Bf16)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data loading.",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA for parameter-efficient fine-tuning.",
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
        "--num_epochs",
        type=int,
        default=200,
        help="Maximum number of epochs to train for.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=20,
        help="Patience for early stopping. Training stops if val loss doesn't improve for this many epochs.",
    )
    args = parser.parse_args()
    main(args)
