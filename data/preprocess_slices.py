import os
import json
import argparse
from tqdm import tqdm

import nibabel as nib
import numpy as np
from PIL import Image


def get_bounding_box(mask: np.ndarray) -> list[int] | None:
    """
    Calculates the tightest bounding box around the 1s in a binary mask

    Args:
        mask (np.ndarray): A 2D NumPy array representing a binary mask

    Returns:
        list | None: A list of 4 integers [x_min, y_min, x_max, y_max]
                      or None if the mask is empty
    """
    rows, cols = np.where(mask > 0)
    if rows.size == 0 or cols.size == 0:
        return None

    x_min = int(np.min(cols))
    y_min = int(np.min(rows))
    x_max = int(np.max(cols))
    y_max = int(np.max(rows))

    return [x_min, y_min, x_max, y_max]


def find_best_slice(mask_volume, tumor_labels=[1, 2, 4]):
    """
    Find the slice index with the largest tumor area

    Args:
        mask_volume: 3D numpy array (H, W, Z)
        tumor_labels: list of integer labels representing tumor regions

    Returns:
        best_slice_idx: index of the slice with maximum tumor area
        max_area: the tumor area in that slice (in pixels)
    """
    areas = [
        np.isin(mask_volume[:, :, z], tumor_labels).sum()
        for z in range(mask_volume.shape[2])
    ]

    best_slice_idx = np.argmax(areas)
    max_area = areas[best_slice_idx]
    return best_slice_idx, max_area


def normalize_slice(slice_2d):
    """
    Normalize 2D slice to 0-255 range

    Args:
        slice_2d: 2D numpy array

    Returns:
        normalized and scaled slice as uint8
    """
    if slice_2d.max() > slice_2d.min():
        normalized = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
    else:
        normalized = np.zeros_like(slice_2d)  # handle constant slice
    return (normalized * 255).astype(np.uint8)


def process_directory(base_data_dir, output_suffix=""):
    """
    Process a directory of patient data

    Args:
        base_data_dir: path to the base directory containing patient folders
        output_suffix: suffix to add to output directories

    Returns:
        metadata: directory mapping filenames to bounding boxes
        processed_count: number of successfully processed patients
    """

    image_dir = f"./data/images{output_suffix}"
    mask_dir = f"./data/masks{output_suffix}"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    metadata = {}

    print(f"Scanning for patient data in: {base_data_dir}")
    # Get a sorted list of all subdirectories
    patient_folders = sorted(
        [
            d
            for d in os.listdir(base_data_dir)
            if os.path.isdir(os.path.join(base_data_dir, d))
        ]
    )

    if not patient_folders:
        print("Error: No patient subdirectories found in the specified data directory.")
        return metadata, 0

    print(f"Found {len(patient_folders)} patient folders. Starting processing...")
    processed_count = 0

    for patient_id in tqdm(
        patient_folders, desc=f"Processing {output_suffix or 'data'}"
    ):
        patient_dir = os.path.join(base_data_dir, patient_id)

        flair_path = os.path.join(patient_dir, f"{patient_id}-t2f.nii.gz")
        seg_path = os.path.join(patient_dir, f"{patient_id}-seg.nii.gz")

        if not os.path.exists(flair_path) or not os.path.exists(seg_path):
            tqdm.write(f"Warning: Missing t2f or seg file for {patient_id}. Skipping.")
            continue

        try:
            flair_img_3d = nib.loadsave.load(flair_path).get_fdata()
            seg_img_3d = nib.loadsave.load(seg_path).get_fdata()

            assert flair_img_3d.shape == seg_img_3d.shape, (
                "Image and mask shapes do not match"
            )

            best_slice_idx, max_tumor_area = find_best_slice(seg_img_3d)

            if max_tumor_area == 0:
                tqdm.write(f"Warning: No tumor found for {patient_id}. Skipping.")
                continue

            best_image_slice = flair_img_3d[:, :, best_slice_idx]
            best_mask_slice = seg_img_3d[:, :, best_slice_idx]
            best_mask = (best_mask_slice > 0).astype(np.uint8)

            bbox = get_bounding_box(best_mask)
            if bbox is None:
                tqdm.write(
                    f"Warning: Could not find bounding box for {patient_id}. Skipping."
                )
                continue

            normalized_slice = normalize_slice(best_image_slice)

            patient_id = os.path.basename(flair_path).split("-t2f.nii.gz")[0]

            img_to_save = Image.fromarray(normalized_slice).convert("RGB")
            image_filename = f"{patient_id}_slice_{best_slice_idx:03d}.png"
            img_to_save.save(os.path.join(image_dir, image_filename))

            mask_to_save = Image.fromarray(best_mask)
            mask_filename = f"{patient_id}_mask_{best_slice_idx:03d}.png"
            mask_to_save.save(os.path.join(mask_dir, mask_filename))

            metadata[image_filename] = bbox

            processed_count += 1

        except Exception as e:
            tqdm.write(f"Error processing {patient_id}: {e}. Skipping.")

    return metadata, processed_count


def main(args):
    print("\n=== Processing Training Data ===")
    train_metadata, train_count = process_directory(
        args.train_data_dir, output_suffix="_train"
    )

    print("\n=== Processing Test Data ===")
    test_metadata, test_count = process_directory(
        args.test_data_dir, output_suffix="_test"
    )

    train_metadata_path = os.path.join("./data", "metadata_train.json")
    test_metadata_path = os.path.join("./data", "metadata_test.json")

    with open(train_metadata_path, "w") as f:
        json.dump(train_metadata, f, indent=4)

    with open(test_metadata_path, "w") as f:
        json.dump(test_metadata, f, indent=4)

    print("\n=== All Finished! ===")
    print(f"Training: Successfully processed {train_count}/1350 patients")
    print(f"Test: Successfully processed {test_count}/271 patients")
    print(f"Training metadata saved to: {train_metadata_path}")
    print(f"Test metadata saved to: {test_metadata_path}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Extract the best slice from NIfTI files for train and test sets"
    )
    arg_parser.add_argument(
        "--train_data_dir",
        type=str,
        default="./training_data1_v2",
        help="Path to the training data directory (1,350 cases)",
    )
    arg_parser.add_argument(
        "--test_data_dir",
        type=str,
        default="./training_data_additional",
        help="Path to the test data directory (271 cases)",
    )
    args = arg_parser.parse_args()
    main(args)
