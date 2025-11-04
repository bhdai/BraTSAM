## Fine-Tuning the Segment Anything Model (SAM) for Brain Tumor Segmentation

This project explores the adaptation of the Segment Anything Model (SAM), a powerful foundation model for image segmentation, to the specialized medical domain of brain tumor segmentation. The primary goal was to develop a robust, heigh-performance model capable of delineating glioma regions from MRI scans, using data from BraTS 2024 challenge (GLI)

The methodology involved two key stages:

1. Fine-tuning SAM using advanced, parameter-efficient techniques to create a powerful, prompt-based segmentation model.
2. Developing a fully automatic pipeline by training a SOTA object detector (YOLOv8m) to generate prompts for the fine-tuned SAM.

The performance of the final pipeline was benchmarked against nnU-Net, a highly respected and considered as SOTA for medical segmentation tasks.

### Dataset

I utilized data from the BraTS 2024 (GLI folder) Challenge. The dataset consists of multi-modal 3D MRI scans (T1, T1-Gd, T2, T2-FLAIR) and corresponding ground truth segmentation masks for glioma.

For 2D approach, I preprocessed the 3D volumes by:

1. Selecting the single 2D axial slice with the largest tumor area from each patient's T2-FLAIR scan.
2. Generating a tight bounding box around the tumor on that slice to serve as a prompt for training.

This resulted in a high-quality dataset of 2D images and corresponding masks, suitable for fine-tuning SAM.

### Fine-Tuning SAM with PEFT and LoRA

The core of the project was to adapt the pre-trained SAM, which was originally trained on natural images, to the specific nuances of brain MRI scans. A naive full fine-tuning of SAM is computationally expensive and risks "catastrophic forgetting." To address this, I adopted a Parameter-Efficient Fine-Tuning (PEFT) strategy using Low-Rank Adaptation (LoRA).

- Through manual inspection of the sam-vit-base architecture, I identified all linear layers within the Vision Encoder and Mask Decoder transformer blocks (qkv, proj, q_proj, k_proj, v_proj, out_proj, lin1, lin2).
- I found that a low rank (r=8) was most effective, aligning with findings in recent literature. This resulted in a model with ~1.35 million trainable parameters (~1.4% of the total).
- To prepare the model for imperfect, real-world prompts, I trained it with bounding box augmentation, randomly perturbing the ground truth box coordinates during training. This proved essential for robust performance in the final pipeline.

### Automatic pipeline with YOLOv8m

To create a fully automated system that does not require manual annotation at inference time, I implemented a two-stage pipeline:

- **Detector**: I fine-tuned a YOLOv8m model on our dataset to act as a highly accurate and efficient "prompt generator." The detector was trained to identify the location of tumors and produce bounding boxes. It achieved an impressive mAP50 of **0.973** on the validation set.
- **Segmenter**: The bounding boxes predicted by YOLOv8 were then fed as prompts to the best fine-tuned, promptable SAM model to produce the final, high-resolution segmentation mask.

### Results

We conducted a series of experiments to evaluate our approach against a formidable nnU-Net baseline and to understand the impact of prompt quality.

| Model / Configuration                         | Test Dice Score | Test IoU Score |
| :-------------------------------------------- | :-------------: | :------------: |
| **nnU-Net Baseline** (trained for 150 epochs) |     0.8687      |     0.8024     |
|                                               |                 |                |
| **SAM + Ground Truth BBox** (perfect prompt)  |   **0.8985**    |   **0.8279**   |
| SAM + GT BBox (10px perturbation)             |     0.8962      |     0.8256     |
| SAM + GT BBox (20px perturbation)             |     0.8879      |     0.8157     |
| SAM (without prompt)                          |     0.8625      |     0.7864     |
|                                               |                 |                |
| **YOLOv8m + SAM** (fully automatic)           |   **0.8817**    |   **0.8092**   |
