from __future__ import annotations
import torch
import torch.nn as nn

class VisualEncoder(nn.Module):
    """
    ViT-B/32 encoder using torchvision.
    Output: [B, 768]
    """
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import vit_b_32, ViT_B_32_Weights

        if pretrained:
            weights = ViT_B_32_Weights.DEFAULT
            model = vit_b_32(weights=weights)
            self.preprocess = weights.transforms()  # Resize/Crop/Normalize recommended by weights
        else:
            model = vit_b_32(weights=None)
            self.preprocess = None

        # torchvision ViT has a final classification head `heads`
        # We want the penultimate embedding.
        model.heads = nn.Identity()
        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] assumed already preprocessed
        return self.model(x)  # [B,768]
