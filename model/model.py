"""
model.py - PSFRegressor (DINOv2 ViT-B/14 + Regression Head)
=============================================================
Wraps the *timm* DINOv2 backbone with a lightweight MLP head that
outputs three continuous PSF scores.

Architecture Overview
---------------------
Backbone : DINOv2 ViT-B/14  (86M params, pretrained self-supervised)
Head     : Linear(768 → 256) → ReLU → Dropout → Linear(256 → 3)
Outputs  : [Creativity, Belonging, EaseOfUse]  (continuous, scale 1–5)

Key helpers
-----------
* freeze_backbone()      - freeze every backbone weight (Phase 1 training).
* unfreeze_last_blocks() - unfreeze the last n transformer blocks (Phase 2).
"""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
#  PSFRegressor
# ──────────────────────────────────────────────────────────────────────────────


class PSFRegressor(nn.Module):
    """
    DINOv2 ViT-B/14 backbone → small regression head → 3 outputs.

    Parameters
    ----------
    backbone_name : str
        ``timm`` model name (default ``vit_base_patch14_dinov2``).
    pretrained : bool
        Load pretrained weights (set to *True* for transfer learning).
    embed_dim : int
        CLS token feature dimension (768 for ViT-B).
    head_hidden : int
        Hidden-layer width of the MLP head.
    dropout : float
        Dropout probability between hidden layers.
    num_outputs : int
        Number of regression targets (default 3: Creativity, Belonging, EaseOfUse).
    super_head : bool
        If True, uses a dual-stream CLS+GAP head with multi-sample dropout.
        The standard architecture (super_head=False) is recommended for
        best generalization based on empirical results (R²=0.42).
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch14_dinov2",
        pretrained: bool = True,
        embed_dim: int = 768,
        head_hidden: int = 256,
        dropout: float = 0.3,
        num_outputs: int = 3,
        super_head: bool = False,
    ):
        super().__init__()
        self.super_head = super_head

        # -- backbone --
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # remove original classifier
            dynamic_img_size=True,
        )

        if super_head:
            # -- Super-Head --
            # Dual stream input: 768 (CLS) + 768 (GAP) = 1536
            input_dim = embed_dim * 2
            self.head = nn.Sequential(
                nn.Linear(input_dim, head_hidden),
                nn.BatchNorm1d(head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.dropout_samples = 5
            self.final_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(self.dropout_samples)])
            self.final_proj = nn.Linear(head_hidden, num_outputs)
        else:
            # -- Standard Architecture (Best Model) --
            # CLS token → MLP → 3 regression outputs
            self.regressor = nn.Sequential(
                nn.Linear(embed_dim, head_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, num_outputs)
            )

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (B, 3, H, W)
            Batch of images pre-processed with ImageNet normalization.

        Returns
        -------
        torch.Tensor  shape (B, num_outputs)
            Predicted PSF scores for each image in the batch.
        """
        # forward_features returns [B, N+1, D] where index 0 is [CLS]
        features = self.backbone.forward_features(x)

        if self.super_head:
            cls_token = features[:, 0]
            patch_tokens = features[:, 1:]
            gap_token = patch_tokens.mean(dim=1)

            # Concatenate for 1536-dim representation
            combined = torch.cat([cls_token, gap_token], dim=1)
            x = self.head(combined)

            # Multi-sample dropout ensemble (training only)
            if self.training:
                out = 0
                for drop in self.final_dropouts:
                    out += self.final_proj(drop(x))
                return out / self.dropout_samples
            else:
                return self.final_proj(x)
        else:
            # Standard: Just CLS token → Regressor
            cls_token = features[:, 0]
            return self.regressor(cls_token)

    # ── freeze / unfreeze helpers ─────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Freeze **all** backbone parameters (Phase 1 training)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("[PSFRegressor] Backbone FROZEN - all params.")

    def unfreeze_last_blocks(self, num_blocks: int = 2) -> None:
        """
        Unfreeze the last *num_blocks* transformer blocks (Phase 2 training).

        Also unfreezes the backbone's final LayerNorm and the regression head.
        """
        # Start fully frozen
        for p in self.backbone.parameters():
            p.requires_grad = False

        total_blocks = len(self.backbone.blocks)
        start_idx = total_blocks - num_blocks

        for idx in range(start_idx, total_blocks):
            for p in self.backbone.blocks[idx].parameters():
                p.requires_grad = True

        # Also unfreeze the final layer-norm
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        # Regression head is always trainable
        if self.super_head:
            for p in self.head.parameters(): p.requires_grad = True
            for p in self.final_proj.parameters(): p.requires_grad = True
        else:
            for p in self.regressor.parameters(): p.requires_grad = True

        total_params     = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100.0 * trainable_params / total_params
        head_type = "Super-Head" if self.super_head else "Standard Head"
        print(
            f"[PSFRegressor] Unfroze last {num_blocks} blocks + {head_type} "
            f"({trainable_params:,} / {total_params:,} params trainable = {pct:.1f}%)"
        )

    # ── convenience ──────────────────────────────────────────────────────────

    def trainable_params_summary(self) -> dict:
        """Return a dict with total, trainable, and frozen parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ──────────────────────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────────────────────


def build_model(cfg: dict) -> PSFRegressor:
    """Construct a ``PSFRegressor`` from a parsed YAML config dict."""
    mc = cfg.get("model", {})
    return PSFRegressor(
        backbone_name=mc.get("backbone", "vit_base_patch14_dinov2"),
        pretrained=mc.get("pretrained", True),
        embed_dim=mc.get("embed_dim", 768),
        head_hidden=mc.get("head_hidden", 256),
        dropout=mc.get("dropout", 0.3),
        num_outputs=mc.get("num_outputs", 3),
        super_head=mc.get("super_head", False),
    )
