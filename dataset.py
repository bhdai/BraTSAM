import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import SamProcessor


class BrainTumorDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        metadata,
        filenames,
        processor,
        perturbation_level=10,
        no_prompt_mode=False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.perturbation_level = perturbation_level
        self.image_filenames = filenames
        self.metadata = metadata
        self.no_prompt_mode = no_prompt_mode

    def __len__(self):
        return len(self.image_filenames)

    def _get_prompt_bbox(self, gt_bbox, image_size):
        """
        generate bounding box prompt and apply random noise

        Args:
            gt_bbox: the ground truth bounding box [x_min, y_min, x_max, y_max]
            image_size: (width, height) of the image

        Returns:
            noisy_bbox: the bounding box with random noise applied
        """
        if self.perturbation_level == 0:
            return gt_bbox

        img_width, img_height = image_size
        x_min, y_min, x_max, y_max = gt_bbox

        noise_strength = np.random.randint(0, self.perturbation_level + 1)

        # generate random noise for each coordinate
        x_min_noise = np.random.randint(-noise_strength, noise_strength + 1)
        y_min_noise = np.random.randint(-noise_strength, noise_strength + 1)
        x_max_noise = np.random.randint(-noise_strength, noise_strength + 1)
        y_max_noise = np.random.randint(-noise_strength, noise_strength + 1)

        # apply noise to expand the box
        noised_x_min = x_min + x_min_noise
        noised_y_min = y_min + y_min_noise
        noised_x_max = x_max + x_max_noise
        noised_y_max = y_max + y_max_noise

        # clamp the coordinates to stay within image boundaries
        final_x_min = max(0, noised_x_min)
        final_y_min = max(0, noised_y_min)
        final_x_max = min(img_width - 1, noised_x_max)
        final_y_max = min(img_height - 1, noised_y_max)

        # ensure min < max
        if final_x_min >= final_x_max:
            final_x_max = final_x_min + 1
        if final_y_min >= final_y_max:
            final_y_max = final_y_min + 1

        return [final_x_min, final_y_min, final_x_max, final_y_max]

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        mask_filename = image_filename.replace("_slice_", "_mask_")

        image_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = Image.open(image_path).convert("RGB")

        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.float32)

        if self.no_prompt_mode:
            # Use full image as bbox prompt for prompt-free inference
            img_width, img_height = image.size
            full_image_bbox = [0, 0, img_width, img_height]
            inputs = self.processor(
                image,
                input_boxes=[[full_image_bbox]],
                segmentation_maps=mask,
                return_tensors="pt",
            )
        else:
            gt_bbox = self.metadata[image_filename]
            prompt_bbox = self._get_prompt_bbox(gt_bbox, image.size)

            inputs = self.processor(
                image,
                input_boxes=[[prompt_bbox]],
                segmentation_maps=mask,
                return_tensors="pt",
            )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs
