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

            # LoRA is most effective on the q and v projections in the attention layers
            # fmt: off
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=[
                    "qkv", "proj", # for Vision Encoder
                    "q_proj", "k_proj", "v_proj", "out_proj", # for Mask Decoder Attention
                    "lin1", "lin2" # for MLP blocks in both
                ],
                lora_dropout=0.1,
                bias="none",
            )
            # fmt: on

            # wrap the model with peft
            self.model = get_peft_model(self.model, lora_config)

            print("LoRA applied. Trainable parameters:")
            self.model.print_trainable_parameters()
        else:
            print("Using vanilla fine-tuning (unfreezing specific layers).")
            for name, param in self.model.named_parameters():
                if name.startswith("vision_encoder") or name.startswith(
                    "prompt_encoder"
                ):
                    param.requires_grad = False

    def forward(self, pixel_values, input_boxes, full_outputs=False):
        """
        forward pass through the SAM model

        Args:
            pixel_values (torch.Tensor): the preprocessed image tensor
            input_boxes (torch.Tensor): the bounding box prompt tensor
            full_outputs (bool): if True, return dict with mask and IoU scores
                                 for smart mask selection; if False, return only
                                 the first predicted mask (legacy behavior)

        Returns:
            If full_outputs=False: torch.Tensor - predicted masks [:, 0, 0, :, :]
            If full_outputs=True: dict with:
                - 'pred_masks': torch.Tensor [B, N, M, H, W] - all mask proposals
                - 'iou_scores': torch.Tensor [B, N, M] - IoU scores for each mask
        """
        outputs = self.model(
            pixel_values=pixel_values, input_boxes=input_boxes
        )  # [B, N, M, H, W] where N is number of input prompts, M is number of mask proposals (3 by default)

        if full_outputs:
            return {
                'pred_masks': outputs.pred_masks,
                'iou_scores': outputs.iou_scores,
            }

        # Legacy behavior: select the first mask
        predicted_masks = outputs.pred_masks[:, 0, 0, :, :]

        return predicted_masks
