import os
import json
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def setup_directories(base_dir):
    """create nn-Unet directory structure"""
    dataset_dir = base_dir / "nnUNet_raw" / "Dataset001_BrainTumorFLAIR"

    dirs = {
        "imagesTr": dataset_dir / "imagesTr",
        "labelsTr": dataset_dir / "labelsTr",
        "imagesTs": dataset_dir / "imagesTs",
        "labelsTs": dataset_dir / "labelsTs",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dataset_dir, dirs


def convert_rgb_to_grayscale(rgb_image_path, output_path):
    """
    Convert RGB PNG which is grayscale replicated 3x back to grayscale

    Args:
        rgb_image_path: Path to RGB PNG file
        output_path: Path to save grayscale PNG
    """
    img = Image.open(rgb_image_path)

    # convert to grayscale all 3 channels are identical anyway
    grayscale_img = img.convert("L")

    grayscale_img.save(output_path)


def process_dataset(
    sam_image_dir, sam_mask_dir, nnunet_image_dir, nnunet_label_dir, split_name
):
    """
    Process images and masks from SAM format to nnU-Net format

    Args:
        sam_image_dir: Directory with SAM RGB images
        sam_mask_dir: Directory with SAM binary masks
        nnunet_image_dir: Output directory for nnU-Net images
        nnunet_label_dir: Output directory for nnU-Net labels
        split_name: 'train' or 'test'

    Returns:
        count: Number of processed cases
    """
    sam_image_dir = Path(sam_image_dir)
    sam_mask_dir = Path(sam_mask_dir)

    if not sam_image_dir.exists():
        print(f"Warning: {sam_image_dir} does not exist. Skipping {split_name} set.")
        return 0

    # Get all image files
    image_files = sorted(list(sam_image_dir.glob("*.png")))

    if not image_files:
        print(f"Warning: No PNG files found in {sam_image_dir}")
        return 0

    print(f"\nProcessing {split_name} set: {len(image_files)} cases")

    processed_count = 0

    for image_path in tqdm(image_files, desc=f"Converting {split_name}"):
        # Extract case identifier from filename
        # Format: {patient_id}_slice_{slice_idx}.png
        filename = image_path.name
        case_id = filename.replace(".png", "")

        # find corresponding mask
        mask_filename = filename.replace("_slice_", "_mask_")
        mask_path = sam_mask_dir / mask_filename

        if not mask_path.exists():
            print(f"Warning: Mask not found for {filename}. Skipping.")
            continue

        # nnU-Net naming convention
        # Images: {case_id}_0000.png (0000 indicates channel/modality 0)
        # Labels: {case_id}.png
        nnunet_image_filename = f"{case_id}_0000.png"
        nnunet_label_filename = f"{case_id}.png"

        nnunet_image_path = nnunet_image_dir / nnunet_image_filename
        nnunet_label_path = nnunet_label_dir / nnunet_label_filename

        # convert RGB image to grayscale
        convert_rgb_to_grayscale(image_path, nnunet_image_path)

        # copy mask directly as it's already in correct format: binary 0/1
        shutil.copy(mask_path, nnunet_label_path)

        processed_count += 1

    return processed_count


def create_dataset_json(dataset_dir, num_training):
    """
    Create the dataset.json file required by nnU-Net

    Args:
        dataset_dir: Path to Dataset001_BrainTumorFLAIR directory
        num_training: Number of training cases
    """
    dataset_json = {
        "channel_names": {"0": "FLAIR"},  # zscore by default
        "labels": {"background": 0, "tumor": 1},
        "numTraining": num_training,
        "file_ending": ".png",
    }

    json_path = dataset_dir / "dataset.json"

    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"\nDataset JSON created at: {json_path}")
    print("Configuration:")
    print("  - Channel: FLAIR (single modality)")
    print("  - Normalization: zscore (default for non-CT)")
    print("  - Labels: background=0, tumor=1")
    print("  - Training cases: {num_training}")


def main():
    base_dir = Path(".")
    sam_data_dir = Path("../data")  # parent directory where SAM data is stored

    print("=" * 60)
    print("nnU-Net Data Preprocessing")
    print("=" * 60)
    print(f"SAM data directory: {sam_data_dir.resolve()}")
    print(f"nnU-Net base directory: {base_dir.resolve()}")

    # setup nnU-Net directory structure
    dataset_dir, dirs = setup_directories(base_dir)
    print(f"\nCreated nnU-Net directory structure at: {dataset_dir}")

    # process training set
    num_training = process_dataset(
        sam_image_dir=sam_data_dir / "images_train",
        sam_mask_dir=sam_data_dir / "masks_train",
        nnunet_image_dir=dirs["imagesTr"],
        nnunet_label_dir=dirs["labelsTr"],
        split_name="train",
    )

    # process test set
    num_test = process_dataset(
        sam_image_dir=sam_data_dir / "images_test",
        sam_mask_dir=sam_data_dir / "masks_test",
        nnunet_image_dir=dirs["imagesTs"],
        nnunet_label_dir=dirs["labelsTs"],
        split_name="test",
    )

    # create dataset.json
    if num_training > 0:
        create_dataset_json(dataset_dir, num_training)
    else:
        print("\nError: No training data was processed. Cannot create dataset.json")
        return

    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Training cases: {num_training}")
    print(f"Test cases: {num_test}")


if __name__ == "__main__":
    main()
