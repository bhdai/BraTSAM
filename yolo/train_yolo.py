from ultralytics import YOLO


def main():
    model = YOLO("yolov8m.pt")

    data_config_path = "./yolo_dataset/data.yaml"

    epochs = 100
    batch_size = 24
    img_size = 256

    print("--- Starting YOLOv8 Fine-Tuning ---")

    results = model.train(
        data=data_config_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=10,
        project="yolo_runs",
        name="braintumor_yolov8m",
        device="cuda",
    )

    print("\n--- YOLOv8 Fine-Tuning Complete! ---")
    print(f"Best model weights saved at: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
