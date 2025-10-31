import torch.nn as nn
from transformers import SamModel
from peft import get_peft_model, LoraConfig, TaskType


class SamFineTuner(nn.Module):
    def __init__(
        self,
        model_id="facebook/sam-vit-base",
        use_lora=False,
        lora_rank=8,
        lora_alpha=16,
        freeze_encoders=True,
    ):
        """
        A pytorch module wrapper for the SAM model to handle fine-tuning

        Args:
            model_id (str): the model id of the SAM model
            use_lora (bool): if true, applies LoRA to the model
            lora_rank (int): the rank for the LoRA matricies
            lora_alpha (int): the alpha scaling factor for LoRA
        """
        super().__init__()
        self.model = SamModel.from_pretrained(model_id)

        if use_lora:
            print("Applying LoRA to SAM model...")

            # LoRa is most effective on the q and v projections in the attention layers
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
            )

            # wrap the model with peft
            self.model = get_peft_model(self.model, lora_config)

            # unfreeze the mask decoder manually, as it's not part the peft wrapping target_modules
            for name, param in self.model.named_parameters():
                if name.startswith("mask_decoder"):
                    param.requires_grad = True

            print("LoRA applied. Trainable parameters:")
            self.model.print_trainable_parameters()
        else:
            print("Using vanilla fine-tuning (unfreezing specific layers).")
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
