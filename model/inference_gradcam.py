"""
inference_gradcam.py - PSF Regression Targeted Interpretability
===============================================================
Uses Attention Rollout (Abnar & Zuidema, 2020) to generate spatial
importance maps for DINOv2 ViT. This is the correct gradient-free
approach for ViTs — standard Grad-CAM fails for frozen ViT backbones
because gradient flow is blocked at the frozen parameter boundary.
"""
import argparse
import datetime
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from dataset import StudioPSFDataset, get_transforms
from model import build_model
from training_script import load_config, set_seed


# ──────────────────────────────────────────────────────────────────────────────
#  Attention Rollout Visualizer
# ──────────────────────────────────────────────────────────────────────────────

class ViTAttentionRollout:
    """
    Attention Rollout for DINOv2 ViT (Abnar & Zuidema, 2020).

    Works by:
    1. Capturing the self-attention weights at every transformer block.
    2. Recursively multiplying them together (with residual connections).
    3. Reading out how much the [CLS] token attends to each spatial patch.

    This works entirely without gradients, making it robust to frozen layers.
    """
    def __init__(self, model, discard_ratio: float = 0.7):
        """
        Args:
            discard_ratio: Fraction of lowest-attention patches to zero out.
                           0.7 means keep the top 30% most attended patches.
        """
        self.model = model
        self.discard_ratio = discard_ratio
        self._attention_maps: list = []
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        """Hook into every attention block to capture softmax weights."""
        for block in self.model.backbone.blocks:
            h = block.attn.register_forward_hook(self._capture_attn)
            self._hooks.append(h)

    def _capture_attn(self, module, inputs, output):
        """
        Recompute attention weights from Q and K (timm ViT stores them).
        We re-derive them here to be independent of timm version quirks.
        """
        # inputs[0] is the token sequence fed into this attention module
        x = inputs[0]  # (B, N, C)
        B, N, C = x.shape
        num_heads = module.num_heads
        head_dim = C // num_heads
        scale = head_dim ** -0.5

        qkv = module.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)  # each (B, H, N, D)

        attn = (q @ k.transpose(-2, -1)) * scale        # (B, H, N, N)
        attn = attn.softmax(dim=-1)                      # (B, H, N, N)

        # Average across heads → (B, N, N)
        self._attention_maps.append(attn.mean(dim=1).detach().cpu())

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """
        Run attention rollout on a single image tensor.

        Args:
            x: Image tensor of shape (1, 3, H, W).

        Returns:
            cam: Float32 numpy array of shape (side, side) with values in [0, 1].
        """
        self._attention_maps.clear()

        # Forward pass — hooks capture attention at every block
        _ = self.model(x)

        if not self._attention_maps:
            side = 37
            return np.ones((side, side), dtype=np.float32) * 0.5

        # Rollout: A_0 · A_1 · ... · A_L  (with residual skip)
        num_tokens = self._attention_maps[0].shape[-1]
        rollout = torch.eye(num_tokens).unsqueeze(0)  # (1, N, N)

        for attn in self._attention_maps:
            # Add 50% identity residual (models every skip connection)
            a = 0.5 * attn + 0.5 * torch.eye(num_tokens).unsqueeze(0)
            # Row-normalize so each row sums to 1
            a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            rollout = a @ rollout  # (1, N, N)

        # CLS token (row 0) → attention to every spatial patch (cols 1:)
        cls_attn = rollout[0, 0, 1:].numpy()  # (1369,)

        # Discard lowest-attention patches (background noise reduction)
        threshold = np.percentile(cls_attn, self.discard_ratio * 100)
        cls_attn = np.where(cls_attn >= threshold, cls_attn, 0.0)

        # Reshape to 2D patch grid
        side = int(np.sqrt(len(cls_attn)))
        cam = cls_attn.reshape(side, side).astype(np.float32)

        # Light Gaussian smoothing to blend patch-boundary edges
        cam = scipy.ndimage.gaussian_filter(cam, sigma=1.0)

        # Normalise to [0, 1]
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min + 1e-8)
        else:
            cam = np.ones((side, side), dtype=np.float32) * 0.5

        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ──────────────────────────────────────────────────────────────────────────────
#  Overlay Utility
# ──────────────────────────────────────────────────────────────────────────────

def overlay_cam_with_text(img: Image.Image, mask: np.ndarray,
                          ai_score: float, gt_score: float,
                          dim_name: str) -> Image.Image:
    """
    Overlay the attention heatmap and append a monospace caption strip below.
    This is the stable version that produced the final_training_2026-04-25_23-00-10 run.
    """
    img_w, img_h = img.size

    # ── 1. Upsample mask to image resolution ─────────────────────────────────
    mask_resized = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    # ── 2. Gamma correction to boost mid-tones ───────────────────────────────
    mask_boosted = np.power(mask_resized, 0.4)
    m_min, m_max = mask_boosted.min(), mask_boosted.max()
    if m_max > m_min:
        mask_boosted = (mask_boosted - m_min) / (m_max - m_min + 1e-8)

    # ── 3. JET colormap ───────────────────────────────────────────────────────
    mask_uint8 = np.uint8(255 * mask_boosted)
    heatmap_bgr = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # ── 4. Selective alpha: Tightened blue radius by 40% ─────────────────────
    # We increase the cutoff so the blues become transparent sooner,
    # shrinking the "radius" of the low-attention halo.
    BLUE_CUTOFF = 0.45
    alpha_map = np.where(
        mask_boosted < BLUE_CUTOFF,
        ((mask_boosted / BLUE_CUTOFF) ** 2) * 0.52,  # Quadratic ramp for tighter edges
        0.52                                          # Flat for green, yellow, red
    ).astype(np.float32)
    alpha_3ch = np.stack([alpha_map] * 3, axis=-1)

    img_np = np.float32(img)
    composite = np.float32(heatmap_rgb) * alpha_3ch + img_np * (1.0 - alpha_3ch)
    result_img = Image.fromarray(np.uint8(np.clip(composite, 0, 255)))


    # ── 5. Caption strip below the image ─────────────────────────────────────
    STRIP_H = 42
    canvas_h = img_h + STRIP_H
    canvas = Image.new("RGB", (img_w, canvas_h), color=(15, 15, 15))
    canvas.paste(result_img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    # JetBrains Mono → Courier New → system default
    font_size = 18
    font_paths = [
        r"C:\Windows\Fonts\JetBrainsMono-Regular.ttf",
        r"C:\Windows\Fonts\JetBrainsMonoNL-Regular.ttf",
        "cour.ttf",
        "DejaVuSansMono.ttf",
    ]
    font = None
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    gap = abs(ai_score - gt_score)
    status = "SUCCESS" if gap <= 0.5 else "FAIL"
    caption = f"[{dim_name.upper():<10}] ViT: {ai_score:.2f} | GT: {gt_score:.2f} | Gap: {gap:.2f} ({status})"
    text_y = img_h + (STRIP_H - font_size) // 2
    draw.text((14, text_y), caption, fill=(220, 220, 220), font=font)

    return canvas








# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Checkpoint and its internal config
    ckpt = torch.load(args.ckpt, map_location=dev)
    if "config" in ckpt:
        print(f"[info] Using configuration found in checkpoint: {args.ckpt}")
        cfg = ckpt["config"]
    else:
        print(f"[warning] No config in checkpoint. Falling back to CLI config: {args.config}")
        cfg = load_config(args.config)

    set_seed(cfg["train"].get("seed", 2026))
    
    model = build_model(cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dc = cfg["data"]
    root = Path(dc["root"])
    img_size = dc.get("img_size", 518)
    cols = dc.get("label_cols", ["Creativity", "Belonging", "EaseOfUse"])

    dataset = StudioPSFDataset(
        root_dir=root,
        image_dir=dc.get("image_dir", "Renamed Images"),
        labels_csv=dc.get("labels_csv", "labels.csv"),
        image_col=dc.get("image_col", "Name"),
        label_cols=cols,
        transform=get_transforms(img_size, False),
    )

    # Load the most recent predictions.csv to find best/worst predictions
    eval_dirs = list(Path("evaluation_results").glob("*/predictions.csv"))
    if not eval_dirs:
        print("[error] No predictions.csv found in evaluation_results. Please run evaluate.py first.")
        return
    latest_pred_csv = max(eval_dirs, key=os.path.getmtime)
    print(f"\n[info] Using predictions from: {latest_pred_csv}")
    pred_df = pd.read_csv(latest_pred_csv)

    # Select extreme-case images based on prediction gap
    indices_to_run = []
    
    for col in cols:
        gap_col = f"Gap_{col}"
        pred_df[gap_col] = (pred_df[f"Pred_{col}"] - pred_df[f"GT_{col}"]).abs()
        
        # Closest 3 (Best predictions)
        closest_names = pred_df.sort_values(gap_col).head(3)["Name"].tolist()
        indices_to_run.extend(closest_names)
        
        # Furthest 3 (Worst predictions)
        furthest_names = pred_df.sort_values(gap_col).tail(3)["Name"].tolist()
        indices_to_run.extend(furthest_names)

    selected_names = list(dict.fromkeys(indices_to_run))
    
    # We need to map 'selected_names' back to dataset indices
    # We can create a name-to-index mapping from the dataset
    name_to_idx = {name: idx for idx, name in enumerate(dataset.image_names) if name in selected_names}
    indices = [name_to_idx[n] for n in selected_names if n in name_to_idx]

    pipeline_name = Path(args.config).stem
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(f"gradcam_results/{pipeline_name}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout = ViTAttentionRollout(model, discard_ratio=0.7)

    for idx in tqdm(indices, desc="Attention Rollout"):
        img_tensor, label, name = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(dev)
        img_path = dataset._resolve_image_path(name)
        raw_img = Image.open(img_path).convert("RGB").resize((img_size, img_size))

        with torch.no_grad():
            preds = model(img_tensor).cpu().numpy()[0]

        # One attention map per image (rollout is dimension-agnostic)
        mask = rollout(img_tensor)

        for i, col in enumerate(cols):
            res = overlay_cam_with_text(raw_img, mask, preds[i], label[i].item(), col)
            save_name = f"{name}_{col.lower()}_gradcam.png"
            res.save(out_dir / save_name)

    rollout.remove_hooks()
    print(f"\n[done] Attention Rollout complete. Results: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args())
