#!/usr/bin/env python3
"""
Plot MSE (L2) error comparison across all physics components from epoch_losses.json file.

This script reads the epoch losses JSON file and generates a single plot comparing
MSE metrics for Laplace, Dirichlet, and Neumann components across all epochs.
"""

import json
import matplotlib.pyplot as plt
import argparse
import os
import sys


def plot_mse_comparison(json_file, batch_size, learning_rate, pixel_res, output_dir=None, log_scale=False):
    """
    Plot MSE error comparison across all physics components from epoch_losses.json file.
    
    Args:
        json_file: path to epoch_losses.json file
        batch_size: batch size used in training
        learning_rate: learning rate used in training
        pixel_res: number of pixels (grid resolution)
        output_dir: output directory for saving plots (default: same as json_file)
        log_scale: use logarithmic scale for y-axis (default: False)
    """
    # Load the epoch losses
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    epoch_losses = data['epoch_losses']
    
    # Extract epoch numbers and MSE metrics
    epochs = [loss['epoch'] for loss in epoch_losses]
    laplace_mse = [loss['laplace_loss_mse'] for loss in epoch_losses]
    dirichlet_mse = [loss['dirichlet_loss_mse'] for loss in epoch_losses]
    neumann_mse = [loss['neumann_scaled_loss_mse'] for loss in epoch_losses]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot title with hyperparameters
    scale_suffix = ' (log scale)' if log_scale else ''
    title = f'MSE Error Comparison (L2): BS: {batch_size}; LR: {learning_rate:.1e}; Pixel resolution: {pixel_res}x{pixel_res}{scale_suffix}'
    
    # Plot all MSE metrics on same plot
    ax.plot(epochs, laplace_mse, label='Laplace MSE (∇²u = 0)', marker='o', markersize=3, linewidth=1.5, color='#1f77b4')
    ax.plot(epochs, dirichlet_mse, label='Dirichlet MSE (u = g)', marker='s', markersize=3, linewidth=1.5, color='#ff7f0e')
    ax.plot(epochs, neumann_mse, label='Neumann MSE (∂u/∂n = h, scaled)', marker='^', markersize=3, linewidth=1.5, color='#2ca02c')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE (mean squared error)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Determine output directory and save figure
    if output_dir is None:
        output_dir = os.path.dirname(json_file)
    
    output_suffix = '_log' if log_scale else ''
    output_file = os.path.join(output_dir, f'mse_comparison{output_suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()


def main():
    """Main function with command-line argument support."""
    
    parser = argparse.ArgumentParser(description='Plot MSE error comparison from epoch_losses.json')
    
    parser.add_argument('json_file', type=str,
                        help='Path to epoch_losses.json file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size used in training (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate used in training (default: 1e-3)')
    parser.add_argument('--pixel_res', type=int, default=64,
                        help='Number of pixels (grid resolution) (default: 64)')
    parser.add_argument('--log_scale', action='store_true',
                        help='Use logarithmic scale for y-axis')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Output directory for saving plots (default: same as json_file)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.json_file):
        print(f"Error: File not found: {args.json_file}")
        sys.exit(1)
    
    # Plot the comparison
    plot_mse_comparison(
        json_file=args.json_file,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pixel_res=args.pixel_res,
        output_dir=args.output_dir,
        log_scale=args.log_scale
    )


if __name__ == "__main__":
    main()
