import os
import json
import shutil
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image


def convert_bbox_to_yolo(img_size, bbox):
    """convert bbox from [x_min, y_min, x_max, y_max] to [x_center, y_center, width, height] normalized by image size

    Args:
        img_size: image height and width
        bbox: bouding box in [x_min, y_min, x_max, y_max] format
    """
    img_width, img_height = img_size
    x_min, y_min, x_max, y_max = bbox

    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return x_center, y_center, width, height


def main():
    source_dir = "../data/images_train"
    source_metadata_dir = "../data/metadata_train.json"

    out_dir = "./yolo_dataset"
    val_split = 0.15

    with open(source_metadata_dir, "r") as f:
        metadata = json.load(f)

    all_filenames = list(metadata.keys())
    print(f"Found {len(all_filenames)} total images.")

    train_files, val_files = train_test_split(
        all_filenames, test_size=val_split, random_state=42
    )
    print(f"Splitting data: {len(train_files)} training, {len(val_files)} validation.")

    os.makedirs(os.path.join(out_dir, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels/val"), exist_ok=True)

    splits = {"train": train_files, "val": val_files}

    for split, filenames in splits.items():
        print(f"\nProcessing {split} split...")
        for filename in tqdm(filenames, desc=f"Creating {split} data"):
            source_image_path = os.path.join(source_dir, filename)
            dest_image_path = os.path.join(out_dir, "images", split, filename)
            shutil.copy(source_image_path, dest_image_path)

            bbox = metadata[filename]

            with Image.open(source_image_path) as img:
                img_size = img.size  # (width, height)

            yolo_bbox = convert_bbox_to_yolo(img_size, bbox)

            label_filename = os.path.splitext(filename)[0] + ".txt"
            dest_label_path = os.path.join(out_dir, "labels", split, label_filename)

            with open(dest_label_path, "w") as f:
                # class index is 0 for "tumor"
                f.write(
                    f"0 {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n"
                )

    data_yaml = {
        "path": os.path.abspath(out_dir),  # ultralytics recommends absolute paths
        "train": "images/train",
        "val": "images/val",
        "names": {0: "tumor"},
    }

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"\nCreated {yaml_path} successfully.")
    print("\n--- YOLO Data Preparation Complete! ---")
    print(f"Dataset is ready at: {out_dir}")


if __name__ == "__main__":
    main()
