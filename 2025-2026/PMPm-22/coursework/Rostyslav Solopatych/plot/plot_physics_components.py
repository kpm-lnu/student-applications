#!/usr/bin/env python3
"""
Plot physics loss components (Laplace, Dirichlet, Neumann) from epoch_losses.json file.

This script reads the epoch losses JSON file and generates plots showing
the individual physics loss components across all epochs.
"""

import json
import matplotlib.pyplot as plt
import argparse
import os
import sys


def plot_physics_components(json_file, batch_size, learning_rate, pixel_res, output_dir=None):
    """
    Plot physics loss components (Laplace, Dirichlet, Neumann) from epoch_losses.json file.
    
    Args:
        json_file: path to epoch_losses.json file
        batch_size: batch size used in training
        learning_rate: learning rate used in training
        pixel_res: number of pixels (grid resolution)
        output_dir: output directory for saving plots (default: same as json_file)
    """
    # Load the epoch losses
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    epoch_losses = data['epoch_losses']
    
    # Extract epoch numbers and component losses
    epochs = [loss['epoch'] for loss in epoch_losses]
    laplace_losses = [loss['laplace_loss_mse'] for loss in epoch_losses]
    dirichlet_losses = [loss['dirichlet_loss_mse'] for loss in epoch_losses]
    neumann_losses = [loss['neumann_scaled_loss_mse'] for loss in epoch_losses]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot title with hyperparameters
    suptitle = f'Physics Loss Components: BS: {batch_size}; LR: {learning_rate:.1e}; Pixel resolution: {pixel_res}x{pixel_res}'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    
    # Plot Laplace loss
    axes[0].plot(epochs, laplace_losses, label='Laplace Loss MSE', marker='o', markersize=3, linewidth=1.5, color='#1f77b4')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Laplace Loss MSE', fontsize=11)
    axes[0].set_title('Laplace Equation (∇²u = 0)', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Dirichlet loss
    axes[1].plot(epochs, dirichlet_losses, label='Dirichlet Loss MSE', marker='s', markersize=3, linewidth=1.5, color='#ff7f0e')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Dirichlet Loss MSE', fontsize=11)
    axes[1].set_title('Dirichlet BC (u = g)', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Plot Neumann loss (scaled)
    axes[2].plot(epochs, neumann_losses, label='Neumann Loss MSE (scaled)', marker='^', markersize=3, linewidth=1.5, color='#2ca02c')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Neumann Loss MSE (scaled)', fontsize=11)
    axes[2].set_title('Neumann BC (∂u/∂n = h)', fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Determine output directory and save figure
    if output_dir is None:
        output_dir = os.path.dirname(json_file)
    
    output_file = os.path.join(output_dir, 'physics_components.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()


def main():
    """Main function with command-line argument support."""
    
    parser = argparse.ArgumentParser(description='Plot physics loss components from epoch_losses.json')
    
    parser.add_argument('json_file', type=str,
                        help='Path to epoch_losses.json file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size used in training (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate used in training (default: 1e-3)')
    parser.add_argument('--pixel_res', type=int, default=64,
                        help='Number of pixels (grid resolution) (default: 64)')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Output directory for saving plots (default: same as json_file)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.json_file):
        print(f"Error: File not found: {args.json_file}")
        sys.exit(1)
    
    # Plot the components
    plot_physics_components(
        json_file=args.json_file,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pixel_res=args.pixel_res,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
