"""
evaluate.py - PSF Regression Evaluation
=======================================
Computes hold-out metrics and generates professional visualizations
including Density Scatters, Radar Achievement, and Bland-Altman plots.

CUDA note:
  Evaluation runs on GPU (CUDA) when available. If PyTorch reports no CUDA,
  install the CUDA-enabled build:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  Then re-run this script.  CPU fallback is used automatically when CUDA is absent.
"""
import argparse
import datetime
import json
import os
from pathlib import Path

import matplotlib
# Keep SVG text as real text (not outlines) so figures are editable in
# Illustrator / Inkscape and text is searchable in the final PDF thesis.
matplotlib.rcParams['svg.fonttype'] = 'none'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import torch
from scipy import stats
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import StudioPSFDataset, get_transforms
from model import build_model
from training_script import load_config, set_seed


def find_best_checkpoint(base_dir: str = "checkpoints") -> tuple:
    """
    Scan all sub-directories of *base_dir* for a ``best.pt`` file.
    For each candidate, read the accompanying ``training_history.csv``
    (if present) and record the minimum ``val_mae``.

    Returns
    -------
    (str, float | None)
        Absolute path to the best checkpoint and its val_mae score
        (or ``None`` when no history CSV was found).
    """
    base = Path(base_dir)
    candidates = []  # list of (val_mae, Path)

    if not base.exists():
        fallback = base / "best.pt"
        print(f"[warn] Checkpoint directory '{base}' not found; defaulting to {fallback}")
        return str(fallback), None

    # Collect every directory that contains a best.pt
    search_dirs = [base] + [d for d in sorted(base.iterdir()) if d.is_dir()]
    for ckpt_dir in search_dirs:
        best_pt = ckpt_dir / "best.pt"
        if not best_pt.exists():
            continue
        history_csv = ckpt_dir / "training_history.csv"
        if history_csv.exists():
            try:
                df = pd.read_csv(history_csv)
                if 'val_mae' in df.columns:
                    min_val_mae = float(df['val_mae'].min())
                    candidates.append((min_val_mae, best_pt))
                    continue
            except Exception:
                pass
        # No history or unreadable — add with infinity so it ranks last
        candidates.append((float('inf'), best_pt))

    if not candidates:
        fallback = base / "best.pt"
        print(f"[warn] No best.pt found under '{base}'; defaulting to {fallback}")
        return str(fallback), None

    candidates.sort(key=lambda x: x[0])
    best_mae, best_path = candidates[0]
    print(f"[auto] Best checkpoint: {best_path}  (val_mae={best_mae:.4f})")
    for mae, p in candidates:
        marker = " <-- selected" if p == best_path else ""
        print(f"       {mae:.4f}  {p}{marker}")
    return str(best_path), best_mae

def save_figure(fig, out_path, **kwargs):
    """Save figure in both PNG and SVG formats.

    SVG files have ``svg.fonttype`` set to ``'none'`` globally so that all
    text elements remain editable strings (not path outlines) in vector
    editors and are searchable in PDFs.
    """
    out_path = Path(out_path)
    base_path = out_path.parent / out_path.stem

    # PNG — raster copy for quick preview / thesis submission
    fig.savefig(str(base_path) + ".png", format='png', **kwargs)

    # SVG — vector copy; strip 'dpi' kwarg because SVG is resolution-independent
    svg_kwargs = {k: v for k, v in kwargs.items() if k != 'dpi'}
    fig.savefig(str(base_path) + ".svg", format='svg', **svg_kwargs)

def plot_density_scatters(gts, preds, cols, out_dir):
    """Generate professional density-colored scatter plots with a shared colorbar."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor='white')
    scatters = []
    for i, col in enumerate(cols):
        ax = axes[i]
        x, y = gts[:, i], preds[:, i]
        xy = np.vstack([x, y]); z = gaussian_kde(xy)(xy)
        idx = z.argsort(); x, y, z = x[idx], y[idx], z[idx]
        scatter = ax.scatter(x, y, c=z, s=35, cmap='viridis', alpha=0.75, edgecolors='white', linewidth=0.5)
        scatters.append(scatter)
        
        # Reference lines with professional styling
        ax.plot([1, 5], [1, 5], color='#555555', linestyle='--', linewidth=2, label='Identity (Ideal)', zorder=3)
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
        ax.plot(x, intercept + slope*x, color='#000000', label=f'Linear Fit ($r$={r_val:.2f})', linewidth=1.8, zorder=4)
        
        # Axis styling
        ax.set_title(f"Density: {col}", fontweight='bold', fontsize=14, color='#000000', pad=10)
        ax.set_xlabel("Human Ground Truth (1-5)", fontweight='bold', fontsize=11, color='#000000')
        ax.set_ylabel("AI Prediction (1-5)", fontweight='bold', fontsize=11, color='#000000')
        ax.set_xlim(0.8, 5.2); ax.set_ylim(0.8, 5.2)
        
        # Grid styling
        ax.grid(True, linestyle='--', alpha=0.35, color='#999999', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Spine styling
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(1)
        ax.tick_params(colors='#000000', labelsize=10)
        
        # Legend styling
        ax.legend(loc='upper left', fontsize=9, frameon=True, shadow=True, fancybox=True, 
                 framealpha=0.96, edgecolor='#cccccc')
        
        # Performance metrics box
        r2 = 1 - (np.sum((x - y)**2) / np.sum((x - np.mean(x))**2))
        mae = np.mean(np.abs(x - y))
        textstr = f"$R^2$: {r2:.2f}\nMAE: {mae:.2f}"
        props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.92, edgecolor='#cccccc', linewidth=1)
        ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=11, verticalalignment='bottom', 
               horizontalalignment='right', bbox=props, fontweight='bold', color='#000000')
    
    # Shared colorbar with professional styling
    cbar = fig.colorbar(scatters[0], ax=axes.ravel().tolist(), pad=0.02, shrink=0.8, aspect=30)
    cbar.set_label('Concentration of Samples\n(Point Density)', fontweight='bold', rotation=270, 
                   labelpad=25, fontsize=11, color='#000000')
    cbar.ax.tick_params(colors='#000000', labelsize=10)
    
    plt.tight_layout()
    save_figure(fig, out_dir / "density_scatters", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def bland_altman_plot(gt, pred, title, save_path):
    """Generate a Bland-Altman plot (mean vs difference) with professional styling."""
    means = (gt + pred) / 2
    diffs = pred - gt
    md, sd = np.mean(diffs), np.std(diffs, axis=0)
    
    fig = plt.figure(figsize=(8, 6), facecolor='white')
    ax = fig.add_subplot(111)
    
    # Scatter points with edge styling
    ax.scatter(means, diffs, alpha=0.6, color='#1f77b4', s=60, edgecolors='white', linewidth=1, zorder=2)
    
    # Mean difference line
    ax.axhline(md, color='#555555', linestyle='--', linewidth=2, label=f'Mean Diff: {md:.2f}', zorder=3)
    
    # Limits of agreement with professional styling
    ax.axhline(md + 1.96 * sd, color='#999999', linestyle=':', linewidth=1.8, label='95% LoA', zorder=3)
    ax.axhline(md - 1.96 * sd, color='#999999', linestyle=':', linewidth=1.8, zorder=3)
    
    # Axis styling
    ax.set_title(f"Bland-Altman: {title}", fontsize=13, fontweight='bold', pad=15, color='#000000')
    ax.set_xlabel("Mean (GT + Pred) / 2", fontsize=11, fontweight='bold', color='#000000')
    ax.set_ylabel("Difference (Pred - GT)", fontsize=11, fontweight='bold', color='#000000')
    
    # Spine styling
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#cccccc')
    ax.spines[['left', 'bottom']].set_linewidth(1)
    ax.tick_params(colors='#000000', labelsize=10)
    
    # Grid styling
    ax.grid(True, linestyle='--', alpha=0.35, color='#999999', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legend styling
    ax.legend(frameon=True, shadow=True, fancybox=True, fontsize=10, framealpha=0.96, 
             edgecolor='#cccccc', facecolor='white')
    
    plt.tight_layout()
    save_figure(fig, save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_training_curves(history_path, out_dir):
    """Plot Loss and MAE curves from training_history.csv with professional styling."""
    if not os.path.exists(history_path): return
    df = pd.read_csv(history_path)
    epochs = df['epoch']
    
    fig = plt.figure(figsize=(12, 5), facecolor='white')
    
    # Professional color palette
    color_train = '#1f77b4'     # Blue
    color_val = '#ff7f0e'       # Orange
    
    # ========== Loss Subplot ==========
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, df['train_loss'], label='Train Loss', color=color_train, marker='o', 
            markersize=4, linewidth=2, zorder=2)
    ax1.plot(epochs, df['val_loss'], label='Val Loss', color=color_val, marker='x', 
            markersize=5, linewidth=2, zorder=2)
    
    ax1.set_title('Training & Validation Loss', fontsize=13, fontweight='bold', pad=12, color='#000000')
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold', color='#000000')
    ax1.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold', color='#000000')
    
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.spines[['left', 'bottom']].set_color('#cccccc')
    ax1.spines[['left', 'bottom']].set_linewidth(1)
    ax1.tick_params(colors='#000000', labelsize=10)
    
    ax1.grid(True, linestyle='--', alpha=0.35, color='#999999', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    ax1.legend(frameon=True, shadow=True, fancybox=True, fontsize=10, framealpha=0.96, 
              edgecolor='#cccccc', facecolor='white')
    
    # ========== MAE Subplot ==========
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, df['train_mae'], label='Train MAE', color='#2ca02c', marker='o', 
            markersize=4, linewidth=2, zorder=2)
    ax2.plot(epochs, df['val_mae'], label='Val MAE', color='#d62728', marker='x', 
            markersize=5, linewidth=2, zorder=2)
    
    ax2.set_title('Training & Validation MAE', fontsize=13, fontweight='bold', pad=12, color='#000000')
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold', color='#000000')
    ax2.set_ylabel('MAE', fontsize=11, fontweight='bold', color='#000000')
    
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.spines[['left', 'bottom']].set_color('#cccccc')
    ax2.spines[['left', 'bottom']].set_linewidth(1)
    ax2.tick_params(colors='#000000', labelsize=10)
    
    ax2.grid(True, linestyle='--', alpha=0.35, color='#999999', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    ax2.legend(frameon=True, shadow=True, fancybox=True, fontsize=10, framealpha=0.96, 
              edgecolor='#cccccc', facecolor='white')
    
    plt.tight_layout()
    save_figure(fig, out_dir / "learning_curves", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_radar_achievement(metrics, out_dir):
    """
    Radar chart suite with professional styling for color and B&W print:
    1. Overall comparison of all dimensions.
    2. Individual charts for each dimension.
    3. Saves raw achievement data to CSV for Excel.
    """
    # Professional color palette that works in grayscale (ColorBrewer-inspired)
    targets = {"MAE": 0.60, "RMSE": 0.80, "R2": 0.40, "Pearson": 0.65, "ICC": 0.70}
    categories = list(targets.keys())
    dims = list(metrics.keys())
    # Colors chosen for printability: distinct in both color and grayscale
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    hatches = ['///', '\\\\\\', '|||']  # Hatching patterns for B&W distinction
    line_styles = ['-', '--', '-.']  # Line styles for additional distinction
    
    # -- Data Preparation for CSV --
    achievement_rows = []
    
    # 1. Overall Chart
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1) 
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]
    
    # Target baseline with professional styling
    ax.plot(angles, [1.0]*(len(categories)+1), color='#555555', linewidth=2.5, linestyle='--', label='Target Baseline (100%)', zorder=10)
    ax.fill(angles, [1.0]*(len(categories)+1), color='#cccccc', alpha=0.12)
    
    overall_max = 1.2
    
    for i, dim in enumerate(dims):
        v = [
            targets["MAE"] / max(metrics[dim]["MAE"], 1e-6),
            targets["RMSE"] / max(metrics[dim]["RMSE"], 1e-6),
            metrics[dim]["R2"] / targets["R2"],
            metrics[dim]["Pearson_R"] / targets["Pearson"],
            metrics[dim]["ICC2k"] / targets["ICC"]
        ]
        
        # Log data for CSV
        achievement_rows.append({
            "Dimension": dim,
            "MAE_Achievement": v[0], "RMSE_Achievement": v[1],
            "R2_Achievement": v[2], "Pearson_Achievement": v[3], "ICC_Achievement": v[4]
        })
        
        v_closed = v + [v[0]]
        overall_max = max(overall_max, max(v_closed))
        # Use solid lines with edge emphasis for better distinction
        ax.plot(angles, v_closed, linewidth=2.8, label=dim, color=colors[i], linestyle=line_styles[i], solid_capstyle='round')
        ax.fill(angles, v_closed, color=colors[i], alpha=0.18, edgecolor=colors[i], linewidth=0.5)
    
    # Professional axis styling
    plt.xticks(angles[:-1], categories, fontweight='bold', fontsize=11, color='#333333')
    ax.set_rlabel_position(45) 
    upper_limit = np.ceil(overall_max * 10) / 10 + 0.1
    plt.yticks([0.5, 0.75, 1.0, 1.25, 1.5], ["50%", "75%", "100%", "125%", "150%"], color="#666666", size=9)
    ax.grid(True, linestyle='--', alpha=0.4, color='#888888', linewidth=0.8)
    ax.set_ylim(0, upper_limit)
    
    plt.title("Overall Model Achievement Profile", size=15, fontweight='bold', pad=40, color='#000000')
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.1), frameon=True, shadow=True, fancybox=True, 
              fontsize=10, framealpha=0.96, edgecolor='#cccccc')
    
    # Add subtle spine styling
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1)
    
    save_figure(fig, out_dir / "radar_overall", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # 2. Individual Charts
    for i, dim in enumerate(dims):
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        v = [
            targets["MAE"] / max(metrics[dim]["MAE"], 1e-6),
            targets["RMSE"] / max(metrics[dim]["RMSE"], 1e-6),
            metrics[dim]["R2"] / targets["R2"],
            metrics[dim]["Pearson_R"] / targets["Pearson"],
            metrics[dim]["ICC2k"] / targets["ICC"]
        ]
        v_closed = v + [v[0]]
        
        # Consistent styling with overall chart
        ax.plot(angles, [1.0]*(len(categories)+1), color='#555555', linewidth=2, linestyle='--', label='Target Baseline', zorder=10)
        ax.fill(angles, [1.0]*(len(categories)+1), color='#cccccc', alpha=0.12)
        ax.plot(angles, v_closed, linewidth=3, color=colors[i], label=f'ViT: {dim}', linestyle=line_styles[i], solid_capstyle='round')
        ax.fill(angles, v_closed, color=colors[i], alpha=0.22, edgecolor=colors[i], linewidth=0.5)
        
        plt.xticks(angles[:-1], categories, fontweight='bold', fontsize=10, color='#333333')
        ax.grid(True, linestyle='--', alpha=0.4, color='#888888', linewidth=0.8)
        plt.yticks(color="#666666", size=9)
        ax.set_ylim(0, max(1.2, max(v_closed) + 0.1))
        
        plt.title(f"Target Achievement: {dim}", size=14, fontweight='bold', pad=30, color='#000000')
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True, shadow=True, fancybox=True, 
                  fontsize=10, framealpha=0.96, edgecolor='#cccccc')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(1)
        
        save_figure(fig, out_dir / f"radar_{dim.lower()}", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    # 3. Save Achievement CSV
    pd.DataFrame(achievement_rows).to_csv(out_dir / "radar_achievement_data.csv", index=False)


def plot_summary_metrics(metrics, out_dir):
    """
    Professional bar chart with styling optimized for color and B&W print.
    Uses professional colors, edge styling, and improved typography.
    """
    dims = list(metrics.keys())
    plt.style.use('default')
    
    # Professional color palette with B&W compatibility
    color_mae = '#1f77b4'      # Blue
    color_rmse = '#ff7f0e'     # Orange
    color_r2 = '#2ca02c'       # Green
    color_pearson = '#d62728'  # Red
    color_icc = '#9467bd'      # Purple
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9.5), facecolor='white')
    x, width, line_w = np.arange(len(dims)), 0.35, 2.0
    
    # ========== LEFT PANEL: Predictive Error ==========
    maes, rmses = [metrics[d]["MAE"] for d in dims], [metrics[d]["RMSE"] for d in dims]
    
    # Create bars with professional edge styling
    b1 = ax1.bar(x - width/2, maes, width, label='MAE', color=color_mae, alpha=0.85, 
                 edgecolor='white', linewidth=2, zorder=2)
    b2 = ax1.bar(x + width/2, rmses, width, label='RMSE', color=color_rmse, alpha=0.85, 
                 edgecolor='white', linewidth=2, zorder=2)
    
    # Target lines with professional styling
    ax1.axhline(0.60, color=color_mae, linestyle='--', linewidth=line_w, label='Target: MAE < 0.60', 
                alpha=0.6, zorder=1)
    ax1.axhline(0.80, color=color_rmse, linestyle=':', linewidth=line_w, label='Target: RMSE < 0.80', 
                alpha=0.6, zorder=1)
    
    # Axis styling
    ax1.set_ylabel('Error Value (Lower is Better)', fontweight='bold', fontsize=12, color='#000000')
    ax1.set_title('Predictive Error by Dimension', fontweight='bold', fontsize=14, color='#000000', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dims, fontweight='bold', fontsize=11, color='#000000')
    ax1.set_ylim(0, max(rmses)*1.5)
    
    # Remove top and right spines for cleaner look
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.spines[['left', 'bottom']].set_color('#cccccc')
    ax1.spines[['left', 'bottom']].set_linewidth(1)
    
    # Grid styling
    ax1.grid(True, axis='y', linestyle='--', alpha=0.35, color='#999999', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Legend styling
    ax1.legend(loc='upper right', frameon=True, shadow=True, fancybox=True, fontsize=10, 
               framealpha=0.96, edgecolor='#cccccc', facecolor='white')
    
    # Add value labels and in-bar text
    for label, bars, text_color in zip(['MAE', 'RMSE'], [b1, b2], [color_mae, color_rmse]):
        for bar in bars:
            h = bar.get_height()
            # Top label with value
            ax1.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), 
                        textcoords="offset points", ha='center', va='bottom', fontweight='bold', 
                        fontsize=11, color='#000000')
            # In-bar label
            ax1.text(bar.get_x() + bar.get_width()/2, h/2, label, ha='center', va='center', 
                    rotation=90, color='white', fontweight='bold', fontsize=16, zorder=3)
    
    # ========== RIGHT PANEL: Agreement & Reliability ==========
    r2s = [metrics[d]["R2"] for d in dims]
    prs = [metrics[d]["Pearson_R"] for d in dims]
    iccs = [metrics[d]["ICC2k"] for d in dims]
    
    # Create bars with professional edge styling
    b3 = ax2.bar(x - 0.3, r2s, 0.25, label='$R^2$', color=color_r2, alpha=0.85, 
                 edgecolor='white', linewidth=2, zorder=2)
    b4 = ax2.bar(x, prs, 0.25, label='Pearson $r$', color=color_pearson, alpha=0.85, 
                 edgecolor='white', linewidth=2, zorder=2)
    b5 = ax2.bar(x + 0.3, iccs, 0.25, label='ICC(2,k)', color=color_icc, alpha=0.85, 
                 edgecolor='white', linewidth=2, zorder=2)
    
    # Target lines with professional styling
    ax2.axhline(0.40, color=color_r2, linestyle='--', linewidth=line_w, label='Target: $R^2 > 0.40$', 
                alpha=0.6, zorder=1)
    ax2.axhline(0.65, color=color_pearson, linestyle='-.', linewidth=line_w, label='Target: Pearson $r > 0.65$', 
                alpha=0.6, zorder=1)
    ax2.axhline(0.50, color=color_icc, linestyle=':', linewidth=line_w, label='Target: ICC > 0.50', 
                alpha=0.6, zorder=1)
    
    # Axis styling
    ax2.set_ylabel('Metric Score (Higher is Better)', fontweight='bold', fontsize=12, color='#000000')
    ax2.set_title('Agreement & Reliability by Dimension', fontweight='bold', fontsize=14, color='#000000', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dims, fontweight='bold', fontsize=11, color='#000000')
    ax2.set_ylim(0, 1.45)
    
    # Remove top and right spines for cleaner look
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.spines[['left', 'bottom']].set_color('#cccccc')
    ax2.spines[['left', 'bottom']].set_linewidth(1)
    
    # Grid styling
    ax2.grid(True, axis='y', linestyle='--', alpha=0.35, color='#999999', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # Legend styling
    ax2.legend(loc='upper left', frameon=True, shadow=True, fancybox=True, fontsize=10, 
               framealpha=0.96, edgecolor='#cccccc', facecolor='white')
    
    # Add value labels and in-bar text
    for label, bars, text_color in zip(['R2', 'Pearson', 'ICC'], [b3, b4, b5], 
                                        [color_r2, color_pearson, color_icc]):
        for bar in bars:
            h = bar.get_height()
            # Top label with value
            ax2.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), 
                        textcoords="offset points", ha='center', va='bottom', fontweight='bold', 
                        fontsize=10, color='#000000')
            # In-bar label
            if h > 0.15:  # Only add in-bar label if there's enough space
                ax2.text(bar.get_x() + bar.get_width()/2, h/2, label, ha='center', va='center', 
                        rotation=90, color='white', fontweight='bold', fontsize=14, zorder=3)
    
    plt.tight_layout()
    save_figure(fig, out_dir / "summary_metrics", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_connected_dot_bias(gts, preds, cols, out_dir):
    """Connected Dot Plot (Dumbbell Plot) showing the prediction bias by rating tier."""
    bins, tier_labels = [1.0, 2.0, 3.0, 4.0, 5.0], ["1-2 (Low)", "2-3 (Med-Low)", "3-4 (Med-High)", "4-5 (High)"]
    plot_data = []
    for i, col in enumerate(cols):
        gt, pr = gts[:, i], preds[:, i]
        for j in range(len(bins)-1):
            mask = (gt >= bins[j]) & (gt <= bins[j+1])
            if mask.sum() > 0: plot_data.append({'Dimension': col, 'Tier': tier_labels[j], 'GT_Mean': gt[mask].mean(), 'Pred_Mean': pr[mask].mean(), 'Bias': pr[mask].mean() - gt[mask].mean()})
    if not plot_data: return
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(11, 8.5), facecolor='white')
    y_pos = np.arange(len(plot_data))[::-1]
    gt_means, pr_means, labels, biases = [d['GT_Mean'] for d in plot_data], [d['Pred_Mean'] for d in plot_data], [f"{d['Dimension']} | {d['Tier']}" for d in plot_data], [d['Bias'] for d in plot_data]
    
    # Professional color palette
    color_positive = '#d62728'   # Red
    color_negative = '#1f77b4'   # Blue
    color_human = '#2ca02c'      # Green
    color_pred = '#ff7f0e'       # Orange
    
    # Connection lines with directional coloring
    for y, gt, pr, bias in zip(y_pos, gt_means, pr_means, biases):
        color = color_positive if bias > 0 else color_negative
        ax.plot([gt, pr], [y, y], color=color, alpha=0.6, linewidth=3.5, zorder=2, solid_capstyle='round')
    
    # Scatter dots with edge styling
    ax.scatter(gt_means, y_pos, color=color_human, s=180, label='Human (Ground Truth)', zorder=3, 
              edgecolors='white', linewidth=2, marker='o')
    ax.scatter(pr_means, y_pos, color=color_pred, s=180, label='AI (Prediction)', zorder=3, 
              edgecolors='white', linewidth=2, marker='s')
    
    # Axis styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight='bold', color='#000000')
    ax.set_title("Prediction Bias by Dimension & Rating Tier (Regression to the Mean)", 
                fontsize=15, fontweight='bold', pad=20, color='#000000')
    ax.set_xlabel("Rating Value (1 to 5)", fontsize=12, fontweight='bold', color='#000000')
    
    # Spine styling
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['bottom'].set_linewidth(1)
    ax.tick_params(axis='x', colors='#000000', labelsize=10)
    
    # Legend styling
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), 
             fontsize=11, frameon=True, shadow=True, fancybox=True, framealpha=0.96, edgecolor='#cccccc')
    
    # Grid styling
    ax.grid(axis='x', linestyle='--', alpha=0.35, color='#999999', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Bias labels
    for y, gt, pr, bias in zip(y_pos, gt_means, pr_means, biases):
        mid_x = (gt + pr) / 2
        bias_color = color_positive if bias > 0 else color_negative
        ax.text(mid_x, y + 0.28, f"{bias:+.2f}", ha='center', va='bottom', fontsize=10, 
               fontweight='bold', color=bias_color)
    
    plt.tight_layout()
    save_figure(fig, out_dir / "connected_dot_bias", dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def plot_overall_connected_dot_bias(gts, preds, cols, out_dir):
    """
    Generates a suite of Perception Gap charts with professional styling:
    1. Native: Uses actual means from the test set.
    2. Overridden: Uses user-specified benchmark means.
    3. Zoomed: The overridden version zoomed into the 3.0 - 3.8 range.
    """
    # Configuration for the three versions
    versions = [
        {"name": "overall_connected_dot_bias_native.png", "overrides": {}, "xlim": (3.05, 3.6), "xticks": np.arange(3.1, 3.65, 0.1)},
        {"name": "overall_connected_dot_bias_overridden.png", "overrides": {"Creativity": 3.34, "Belonging": 3.27, "EaseOfUse": 3.37}, "xlim": (3.05, 3.6), "xticks": np.arange(3.1, 3.65, 0.1)},
        {"name": "overall_connected_dot_bias_zoomed.png", "overrides": {"Creativity": 3.34, "Belonging": 3.27, "EaseOfUse": 3.37}, "xlim": (3.05, 3.6), "xticks": np.arange(3.1, 3.65, 0.1)}
    ]
    
    plt.rcParams['font.family'] = 'serif'
    # Professional color palette with B&W compatibility
    human_color = '#2ca02c'     # Green
    vit_color = '#d62728'       # Red
    line_color = '#cccccc'      # Light gray
    
    y_pos = np.arange(len(cols))

    for ver in versions:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
        
        # Calculate means
        gt_means = []
        for i, col in enumerate(cols):
            if col in ver["overrides"]:
                gt_means.append(ver["overrides"][col])
            else:
                gt_means.append(gts[:, i].mean())
        pr_means = [preds[:, i].mean() for i in range(len(cols))]

        for y, gt, pr in zip(y_pos, gt_means, pr_means):
            diff = pr - gt
            ax.plot([gt, pr], [y, y], color=line_color, linewidth=4, zorder=2, solid_capstyle='round')
            ax.scatter(gt, y, color=human_color, s=180, edgecolors='white', linewidth=2, 
                      label='Human' if y==0 else "", zorder=3, marker='o')
            ax.scatter(pr, y, color=vit_color, s=180, marker='s', edgecolors='white', linewidth=2, 
                      label='ViT Model' if y==0 else "", zorder=3)
            
            # Horizontal Labeling Logic (Left or Right of dots)
            h_off = 0.008
            if gt < pr:
                # Human on the left, ViT on the right
                ax.text(gt - h_off, y, f"{gt:.2f}", color=human_color, ha='right', va='center', 
                       fontweight='bold', fontsize=10, fontfamily='serif')
                ax.text(pr + h_off, y, f"{pr:.2f}", color=vit_color, ha='left', va='center', 
                       fontweight='bold', fontsize=10, fontfamily='serif')
            else:
                # Human on the right, ViT on the left
                ax.text(gt + h_off, y, f"{gt:.2f}", color=human_color, ha='left', va='center', 
                       fontweight='bold', fontsize=10, fontfamily='serif')
                ax.text(pr - h_off, y, f"{pr:.2f}", color=vit_color, ha='right', va='center', 
                       fontweight='bold', fontsize=10, fontfamily='serif')
            
            mid_x = (gt + pr) / 2
            ax.text(mid_x, y - 0.35, f"Δ={diff:+.2f}", color='#444444', 
                    ha='center', va='center', fontstyle='italic', fontweight='bold', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.92, edgecolor='#cccccc', boxstyle='round,pad=0.25'),
                    fontfamily='serif')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([c.replace('EaseOfUse', 'Ease of Use') for c in cols], fontsize=12, 
                           fontweight='bold', color='#000000', fontfamily='serif')
        ax.set_xticks(ver["xticks"])
        ax.set_xlim(ver["xlim"])
        ax.set_ylim(-0.8, len(cols) - 0.2)
        
        # Axis styling
        title_suffix = "(Native Means)" if "native" in ver["name"] else "(Thesis Benchmarks)"
        ax.set_title(f"Perception Gap Analysis {title_suffix}", fontsize=17, fontweight='bold', 
                    pad=25, color='#000000', fontfamily='serif')
        
        # Grid and spine styling
        ax.grid(axis='x', linestyle='--', alpha=0.35, color='#999999', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#cccccc')
        ax.spines['bottom'].set_linewidth(1)
        
        # Axis labels
        ax.set_xlabel("Perception Mean Score", fontsize=12, labelpad=12, fontweight='bold', 
                     color='#000000', fontfamily='serif')
        ax.tick_params(axis='x', colors='#000000', labelsize=10)
        
        # Legend styling
        ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True, fancybox=True, 
                 framealpha=0.96, edgecolor='#cccccc', borderpad=0.8, facecolor='white')
        
        plt.tight_layout()
        # Extract base name without extension and format
        base_name = Path(ver["name"]).stem
        save_figure(fig, out_dir / base_name, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
    
    plt.rcParams['font.family'] = 'sans-serif'

@torch.no_grad()
def main(args):
    # ------------------------------------------------------------------ device
    if args.device == "cuda":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            print(f"[info] Device      : {dev} — {torch.cuda.get_device_name(0)}")
            print(f"[info] CUDA version: {torch.version.cuda}")
            print(f"[info] VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            # Enable cuDNN auto-tuner for faster convolutions on fixed input size
            torch.backends.cudnn.benchmark = True
        else:
            print("[warning] " + "=" * 60)
            print(f"[warning] CUDA requested but PyTorch build is: {torch.__version__}")
            print("[warning] This is a CPU-only PyTorch build — GPU is NOT used.")
            print("[warning] To enable CUDA, run ONE of the following commands:")
            print("[warning]   # CUDA 12.1 (recommended for RTX / recent GPUs):")
            print("[warning]   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print("[warning]   # CUDA 11.8 (older GPUs):")
            print("[warning]   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("[warning] Falling back to CPU for this run.")
            print("[warning] " + "=" * 60)
            dev = torch.device("cpu")
    else:
        dev = torch.device("cpu")
        print(f"[info] Device: cpu (CUDA disabled by --device cpu)")

    # Load checkpoint and its internal config
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=dev)
    print(f"[info] Loaded checkpoint: {args.ckpt}")
    
    # Load CLI config first
    cfg = load_config(args.config)
    
    # If checkpoint has a config, we merge it
    if "config" in ckpt:
        print(f"[info] Checkpoint contains internal configuration.")
        ckpt_cfg = ckpt["config"]
        
        # We prioritize CLI for 'data' splits to allow evaluating 
        # any model on any split (e.g. forcing a 15% test set).
        # We prioritize Checkpoint for 'model' and 'train' (head_hidden, backbone, etc.)
        # to ensure the architecture matches the weights.
        
        # Merge logic:
        # 1. Start with checkpoint config (truth for model architecture)
        merged_cfg = ckpt_cfg.copy()
        
        # 2. Override data settings from CLI if they differ
        for k, v in cfg.get("data", {}).items():
            if k in ["val_split", "test_split", "root", "labels_csv", "image_dir"]:
                if merged_cfg["data"].get(k) != v:
                    print(f"[info] Overriding data.{k}: {merged_cfg['data'].get(k)} -> {v} (from CLI)")
                    merged_cfg["data"][k] = v
        
        cfg = merged_cfg
    else:
        print(f"[warning] No config in checkpoint. Using CLI config: {args.config}")

    set_seed(cfg["train"].get("seed", 2026))
    dc, tc, root, img_size = cfg["data"], cfg["train"], Path(cfg["data"]["root"]), cfg["data"].get("img_size", 518)
    
    # EXACT Shuffled Split Matching train.py
    full_ds = StudioPSFDataset(root_dir=root, image_dir=dc.get("image_dir", "Renamed Images"), labels_csv=dc.get("labels_csv", "labels.csv"),
        image_col=dc.get("image_col", "Name"), label_cols=dc.get("label_cols", ["Creativity", "Belonging", "EaseOfUse"]), image_ext=dc.get("image_ext", ".png"),
        transform=get_transforms(img_size, False))
    n = len(full_ds)
    n_test = int(n * dc.get("test_split", 0.0))
    n_val = int(n * dc.get("val_split", 0.20))
    n_train = n - n_val - n_test
    idx = list(range(n))
    rng = np.random.RandomState(tc.get("seed", 2026))
    rng.shuffle(idx)
    
    # If test_split > 0, evaluate on test set. Otherwise evaluate on validation set (legacy).
    eval_idx = idx[n_train + n_val:] if n_test > 0 else idx[n_train : n_train + n_val]
    dataset_name = "Test" if n_test > 0 else "Validation"
    print(f"\n[info] Evaluating on {dataset_name} Set ({len(eval_idx)} images)")
    
    eval_ds = StudioPSFDataset(root_dir=root, image_dir=dc.get("image_dir", "Renamed Images"), labels_csv=dc.get("labels_csv", "labels.csv"),
        image_col=dc.get("image_col", "Name"), label_cols=dc.get("label_cols", ["Creativity", "Belonging", "EaseOfUse"]), image_ext=dc.get("image_ext", ".png"),
        transform=get_transforms(img_size, False), indices=eval_idx)
    loader = DataLoader(eval_ds, batch_size=tc.get("batch_size", 16), num_workers=tc.get("num_workers", 0), pin_memory=True)
    
    model = build_model(cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    all_preds, all_gts, all_names = [], [], []
    for imgs, labels, names in tqdm(loader, desc="Inference"):
        imgs = imgs.to(dev); preds = model(imgs)
        all_preds.append(preds.cpu().numpy()); all_gts.append(labels.numpy()); all_names.extend(names)
    preds, gts = np.concatenate(all_preds, axis=0), np.concatenate(all_gts, axis=0)
    pipeline_name = Path(args.config).stem
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cols, out_dir = dc.get("label_cols", ["Creativity", "Belonging", "EaseOfUse"]), Path(f"evaluation_results/{pipeline_name}_{ts}"); out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for i, col in enumerate(cols):
        y_t, y_p = gts[:, i], preds[:, i]
        mae, rmse, r2 = np.mean(np.abs(y_t-y_p)), np.sqrt(np.mean((y_t-y_p)**2)), 1-(np.sum((y_t-y_p)**2)/np.sum((y_t-np.mean(y_t))**2))
        pr, _ = stats.pearsonr(y_t, y_p)
        icc_df = pd.DataFrame([{"img": j, "rater": r, "score": s} for j in range(len(y_t)) for r, s in [("GT", y_t[j]), ("Pred", y_p[j])]])
        icc_res = pg.intraclass_corr(data=icc_df, targets='img', raters='rater', ratings='score')
        icc_v = icc_res.set_index('Type').loc['ICC2k', 'ICC'] if 'ICC2k' in icc_res['Type'].values else icc_res['ICC'].iloc[0]
        metrics[col] = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "Pearson_R": float(pr), "ICC2k": float(icc_v)}
        bland_altman_plot(y_t, y_p, col, out_dir / f"bland_altman_{col.lower()}")
    
    with open(out_dir / "metrics.json", "w") as f: json.dump(metrics, f, indent=4)
    pred_df = pd.DataFrame(columns=["Name"] + [f"GT_{c}" for c in cols] + [f"Pred_{c}" for c in cols])
    pred_df["Name"] = all_names
    for i, c in enumerate(cols): pred_df[f"GT_{c}"], pred_df[f"Pred_{c}"] = gts[:, i], preds[:, i]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    
    plot_summary_metrics(metrics, out_dir); plot_radar_achievement(metrics, out_dir); plot_density_scatters(gts, preds, cols, out_dir)
    plot_connected_dot_bias(gts, preds, cols, out_dir)
    plot_overall_connected_dot_bias(gts, preds, cols, out_dir)
    plot_training_curves(Path(cfg.get("checkpoint", {}).get("dir", "checkpoints")) / "training_history.csv", out_dir)
    print(f"\n[done] Evaluation complete. Results saved to {out_dir}")

if __name__ == "__main__":
    # Auto-detect the best available checkpoint before parsing args so that
    # --ckpt default is pre-populated with the correct path.
    _auto_ckpt, _auto_mae = find_best_checkpoint("checkpoints")

    parser = argparse.ArgumentParser(
        description="PSF Regression Evaluation — generates PNG + SVG charts."
    )
    parser.add_argument(
        "--ckpt",
        default=_auto_ckpt,
        help=f"Path to checkpoint .pt file (auto-detected: {_auto_ckpt})",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config (falls back to config embedded in checkpoint).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device. 'cuda' falls back to CPU automatically if unavailable.",
    )
    main(parser.parse_args())
