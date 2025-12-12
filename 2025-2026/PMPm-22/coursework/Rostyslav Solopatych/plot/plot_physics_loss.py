#!/usr/bin/env python3
"""
Plot total physics loss (train and validation) from epoch_losses.json file.

This script reads the epoch losses JSON file and generates a plot showing
the training and validation total physics losses across all epochs.
"""

import json
import matplotlib.pyplot as plt
import argparse
import os
import sys


def plot_physics_loss(json_file, batch_size, learning_rate, pixel_res, output_dir=None):
    """
    Plot training and validation total physics losses from epoch_losses.json file.
    
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
    
    # Extract epoch numbers and physics losses
    epochs = [loss['epoch'] for loss in epoch_losses]
    train_physics_losses = [loss['train_physics_loss_mse'] for loss in epoch_losses]
    val_physics_losses = [loss['val_physics_loss_mse'] for loss in epoch_losses]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot title with hyperparameters
    title = f'Total Physics Loss: BS: {batch_size}; LR: {learning_rate:.1e}; Pixel resolution: {pixel_res}x{pixel_res}'
    
    # Plot physics losses
    ax.plot(epochs, train_physics_losses, label='Train Physics Loss MSE', marker='o', markersize=3, linewidth=1.5, color='#2ca02c')
    ax.plot(epochs, val_physics_losses, label='Validation Physics Loss MSE', marker='s', markersize=3, linewidth=1.5, color='#d62728')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Physics Loss MSE', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Determine output directory and save figure
    if output_dir is None:
        output_dir = os.path.dirname(json_file)
    
    output_file = os.path.join(output_dir, 'physics_loss.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.show()


def main():
    """Main function with command-line argument support."""
    
    parser = argparse.ArgumentParser(description='Plot total physics losses from epoch_losses.json')
    
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
    
    # Plot the losses
    plot_physics_loss(
        json_file=args.json_file,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        pixel_res=args.pixel_res,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
