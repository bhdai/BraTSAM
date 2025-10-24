import torch.nn as nn
from transformers import SamModel


class SamFineTuner(nn.Module):
    def __init__(self, model_id="facebook/sam-vit-base", freeze_encoders=True):
        """
        A pytorch module wrapper for the SAM model to handle fine-tuning

        Args:
            freeze_encoders (bool): if true, freezes the vision and prompt encoders
        """
        super().__init__()
        self.model = SamModel.from_pretrained(model_id)

        # freeze the vision and prompt encoders
        if freeze_encoders:
            for name, param in self.model.named_parameters():
                if name.startswith("vision_encoder") or name.startswith(
                    "prompt_encoder"
                ):
                    param.requires_grad = False

    def forward(self, pixel_values, input_boxes):
        """
        forward pass through the SAM model

        Args:
            pixel_values (torch.Tensor): the preprocessed image tensor
            input_boxes (torch.Tensor): the bounding box prompt tensor

        Returns:
            torch.Tensor: The predicted masks
        """
        outputs = self.model(
            pixel_values=pixel_values, input_boxes=input_boxes
        )  # [B, N, M, H, W] where N is number of input prompts, M is number of mask proposals (3 by default)

        # select the first mask and add channel dimension
        predicted_masks = outputs.pred_masks[:, 0, 0, :, :]

        return predicted_masks
