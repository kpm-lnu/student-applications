#!/usr/bin/env python3
"""
Plot grid search results from a saved all_results.json file.

This script renders the same 6-panel summary figure that
``model/grid_search_hyperparameters.py`` produces at the end of a run, but
reads the results from disk so the plot can be regenerated on demand without
re-running the search.

Usage:
    python plot/plot_grid_search.py --results ./grid_search_results/all_results.json
    python plot/plot_grid_search.py --results ./grid_search_results/all_results.json \
        --output ./grid_search_results/grid_search_results.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_grid_search_results(results, search_space, best_config_id=None, output_path=None):
    """
    Render the 6-panel grid-search summary figure.

    Args:
        results: list of per-configuration result dicts. Each must contain at
            least: config_id, batch_size, learning_rate, pixel_res,
            best_val_loss, best_epoch.
        search_space: dict with keys 'batch_sizes', 'learning_rates',
            'pixel_res' (the iteration axes used by the search).
        best_config_id: optional config_id of the winning configuration; used
            to highlight its bar in the "Best Epoch per Configuration" panel.
        output_path: optional path to save the figure as PNG. If None, the
            figure is shown interactively.

    Returns:
        matplotlib.figure.Figure: the generated figure.
    """
    if not results:
        logger.warning("No results to visualize")
        return None

    batch_sizes_axis = search_space.get('batch_sizes', sorted({r['batch_size'] for r in results}))
    learning_rates_axis = search_space.get('learning_rates', sorted({r['learning_rate'] for r in results}))
    pixel_res_axis = search_space.get('pixel_res', sorted({r['pixel_res'] for r in results}))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Grid Search Results: Diffusion Model Hyperparameters',
                 fontsize=16, fontweight='bold')

    batch_sizes = [r['batch_size'] for r in results]
    learning_rates = [r['learning_rate'] for r in results]
    best_epochs = [r['best_epoch'] for r in results]

    # 1. Val Loss vs Batch Size
    ax = axes[0, 0]
    for lr in learning_rates_axis:
        for px in pixel_res_axis:
            mask = [(r['learning_rate'] == lr and r['pixel_res'] == px) for r in results]
            bs = [results[i]['batch_size'] for i, m in enumerate(mask) if m]
            vl = [results[i]['best_val_loss'] for i, m in enumerate(mask) if m]
            if bs:
                pairs = sorted(zip(bs, vl))
                bs, vl = [p[0] for p in pairs], [p[1] for p in pairs]
                ax.plot(bs, vl, marker='o', label=f'LR={lr}, PX={px}')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Validation Diffusion Loss')
    ax.set_title('Loss vs Batch Size')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Val Loss vs Learning Rate (log scale)
    ax = axes[0, 1]
    for bs in batch_sizes_axis:
        for px in pixel_res_axis:
            mask = [(r['batch_size'] == bs and r['pixel_res'] == px) for r in results]
            lr = [results[i]['learning_rate'] for i, m in enumerate(mask) if m]
            vl = [results[i]['best_val_loss'] for i, m in enumerate(mask) if m]
            if lr:
                pairs = sorted(zip(lr, vl))
                lr, vl = [p[0] for p in pairs], [p[1] for p in pairs]
                ax.plot(lr, vl, marker='o', label=f'BS={bs}, PX={px}')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Validation Diffusion Loss')
    ax.set_title('Loss vs Learning Rate')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Val Loss vs Sample Size (pixel_res)
    ax = axes[0, 2]
    for bs in batch_sizes_axis:
        for lr in learning_rates_axis:
            mask = [(r['batch_size'] == bs and r['learning_rate'] == lr) for r in results]
            px = [results[i]['pixel_res'] for i, m in enumerate(mask) if m]
            vl = [results[i]['best_val_loss'] for i, m in enumerate(mask) if m]
            if px:
                pairs = sorted(zip(px, vl))
                px, vl = [p[0] for p in pairs], [p[1] for p in pairs]
                ax.plot(px, vl, marker='o', label=f'BS={bs}, LR={lr}')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Validation Diffusion Loss')
    ax.set_title('Loss vs Sample Size')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Best Epoch vs Configuration
    ax = axes[1, 0]
    config_labels = [f"{r['batch_size']}\n{r['learning_rate']:.0e}\n{r['pixel_res']}"
                     for r in results]
    x_pos = np.arange(len(config_labels))
    colors = ['green' if r.get('config_id') == best_config_id else 'blue' for r in results]
    ax.bar(x_pos, best_epochs, color=colors, alpha=0.7)
    ax.set_xlabel('Configuration (BS/LR/SS)')
    ax.set_ylabel('Best Epoch')
    ax.set_title('Best Epoch per Configuration')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Heatmap: Learning Rate vs Batch Size (averaged over sample sizes)
    ax = axes[1, 1]
    unique_bs = sorted(set(batch_sizes))
    unique_lr = sorted(set(learning_rates))
    heatmap_data = np.zeros((len(unique_lr), len(unique_bs)))

    for i, lr in enumerate(unique_lr):
        for j, bs in enumerate(unique_bs):
            losses = [r['best_val_loss'] for r in results
                      if r['learning_rate'] == lr and r['batch_size'] == bs]
            heatmap_data[i, j] = np.mean(losses) if losses else np.nan

    im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(unique_bs)))
    ax.set_yticks(np.arange(len(unique_lr)))
    ax.set_xticklabels(unique_bs)
    ax.set_yticklabels([f'{lr:.0e}' for lr in unique_lr])
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Loss Heatmap (avg over sample sizes)')
    plt.colorbar(im, ax=ax, label='Val Diffusion Loss')

    for i in range(len(unique_lr)):
        for j in range(len(unique_bs)):
            if not np.isnan(heatmap_data[i, j]):
                ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                        ha="center", va="center", color="white", fontsize=8)

    # 6. Top 10 ranking
    ax = axes[1, 2]
    sorted_results = sorted(results, key=lambda x: x['best_val_loss'])
    config_labels_sorted = [r['config_id'] for r in sorted_results[:10]]
    val_losses_sorted = [r['best_val_loss'] for r in sorted_results[:10]]
    colors_sorted = ['gold' if i == 0 else 'silver' if i == 1 else 'chocolate' if i == 2 else 'blue'
                     for i in range(len(config_labels_sorted))]

    y_pos = np.arange(len(config_labels_sorted))
    ax.barh(y_pos, val_losses_sorted, color=colors_sorted, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(config_labels_sorted, fontsize=8)
    ax.set_xlabel('Validation Diffusion Loss')
    ax.set_title('Top 10 Configurations')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to: {output_path}")

    return fig


def load_results_json(json_path):
    """Load and validate a grid-search all_results.json file.

    Returns:
        tuple: (results, search_space, best_config_id)
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Results file not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    search_space = data.get('search_space', {})
    best_config_id = data.get('best_config_id')

    if not results:
        raise ValueError(f"No 'results' entries found in {json_path}")

    return results, search_space, best_config_id


def main():
    parser = argparse.ArgumentParser(
        description='Plot grid search results from a saved all_results.json file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--results', type=str, required=True,
                        help='Path to all_results.json produced by grid_search_hyperparameters.py')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG path (default: <results_dir>/grid_search_results.png)')
    parser.add_argument('--show', action='store_true',
                        help='Show the figure interactively in addition to saving it')

    args = parser.parse_args()

    results, search_space, best_config_id = load_results_json(args.results)

    output_path = args.output
    if output_path is None:
        output_path = Path(args.results).parent / 'grid_search_results.png'

    fig = plot_grid_search_results(
        results=results,
        search_space=search_space,
        best_config_id=best_config_id,
        output_path=output_path,
    )

    if args.show and fig is not None:
        plt.show()
    else:
        plt.close(fig) if fig is not None else None


if __name__ == "__main__":
    sys.exit(main())
