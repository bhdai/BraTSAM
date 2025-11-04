import os
import json
from ultralytics import YOLO
from tqdm import tqdm


def main():
    YOLO_MODEL_PATH = "./yolo_runs/braintumor_yolov8m/weights/best.pt"
    TEST_IMAGE_DIR = "../data/images_test"
    OUTPUT_JSON_PATH = "../data/metadata_yolo_preds.json"

    print(f"--- Loading YOLO model from: {YOLO_MODEL_PATH} ---")
    model = YOLO(YOLO_MODEL_PATH)

    test_filenames = sorted(
        [f for f in os.listdir(TEST_IMAGE_DIR) if f.endswith(".png")]
    )

    print(f"\nFound {len(test_filenames)} images in the test directory.")
    print("Running inference on all test images to generate bounding box metadata...")

    yolo_metadata = {}

    for filename in tqdm(test_filenames, desc="Predicting BBoxes"):
        image_path = os.path.join(TEST_IMAGE_DIR, filename)

        results = model(image_path, verbose=False)

        if len(results[0].boxes) > 0:
            best_box = results[0].boxes[0]

            pred_coords_xyxy = best_box.xyxy[0].cpu().numpy().astype(int)

            yolo_metadata[filename] = pred_coords_xyxy.tolist()
        else:
            print(f"Warning: No tumor detected in {filename}. Skipping this case.")
            continue

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(yolo_metadata, f, indent=4)

    print("\n--- Prediction Complete ---")
    print(f"Successfully generated YOLO predictions for {len(yolo_metadata)} images.")
    print(f"Metadata saved to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
