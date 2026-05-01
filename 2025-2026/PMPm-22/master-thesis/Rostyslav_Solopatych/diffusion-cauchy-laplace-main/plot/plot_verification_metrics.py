#!/usr/bin/env python3
"""
Plot verification metrics vs noise level from verification_results.csv file.

This script reads the verification results CSV file and generates plots showing
how different error metrics (RMSE, NRMSE, L2, L∞) change with respect to noise level.
It creates both a combined plot with all metrics and individual plots for each metric.
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys


def plot_verification_metrics(csv_file, output_dir=None, log_scale=False):
    """
    Plot verification metrics vs noise level from verification_results.csv file.
    
    Args:
        csv_file: path to verification_results.csv file
        output_dir: output directory for saving plots (default: same as csv_file)
        log_scale: use logarithmic scale for y-axis (default: False)
    """
    # Load the verification results
    df = pd.read_csv(csv_file)
    
    # Convert noise_level: '-' becomes 0, others to float
    df['noise_level_numeric'] = df['noise_level'].apply(lambda x: 0.0 if x == '-' else float(x))
    
    # Sort by noise level for proper plotting
    df = df.sort_values('noise_level_numeric')
    
    # Extract data
    noise_levels = df['noise_level_numeric'].values
    rmse = df['rmse'].values
    nrmse = df['nrmse'].values
    l2_error = df['l2_error'].values
    l2_error_relative = df['l2_error_relative'].values
    linf_error = df['linf_error'].values
    linf_error_relative = df['linf_error_relative'].values
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Plot 1: All metrics combined ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    scale_suffix = ' (log scale)' if log_scale else ''
    title = f'Verification Metrics vs Noise Level{scale_suffix}'
    
    ax.plot(noise_levels, rmse, label='RMSE', marker='o', markersize=4, linewidth=1.5, color='#1f77b4')
    ax.plot(noise_levels, nrmse, label='NRMSE', marker='s', markersize=4, linewidth=1.5, color='#ff7f0e')
    ax.plot(noise_levels, l2_error, label='L2 Error', marker='^', markersize=4, linewidth=1.5, color='#2ca02c')
    ax.plot(noise_levels, l2_error_relative, label='L2 Error (Relative)', marker='d', markersize=4, linewidth=1.5, color='#d62728')
    ax.plot(noise_levels, linf_error, label='L∞ Error', marker='v', markersize=4, linewidth=1.5, color='#9467bd')
    ax.plot(noise_levels, linf_error_relative, label='L∞ Error (Relative)', marker='p', markersize=4, linewidth=1.5, color='#8c564b')
    
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Error Metric Value', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    output_suffix = '_log' if log_scale else ''
    output_file = os.path.join(output_dir, f'verification_metrics_combined{output_suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {output_file}")
    plt.show()
    
    # --- Plot 2: RMSE ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(noise_levels, rmse, label='RMSE', marker='o', markersize=5, linewidth=2, color='#1f77b4')
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Root Mean Square Error vs Noise Level', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'verification_rmse{output_suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"RMSE plot saved to: {output_file}")
    plt.show()
    
    # --- Plot 3: NRMSE ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(noise_levels, nrmse, label='NRMSE', marker='s', markersize=5, linewidth=2, color='#ff7f0e')
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('NRMSE', fontsize=12)
    ax.set_title('Normalized Root Mean Square Error vs Noise Level', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'verification_nrmse{output_suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"NRMSE plot saved to: {output_file}")
    plt.show()
    
    # --- Plot 4: L2 Error ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(noise_levels, l2_error, label='L2 Error', marker='^', markersize=5, linewidth=2, color='#2ca02c')
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('L2 Error vs Noise Level', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'verification_l2_error{output_suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"L2 Error plot saved to: {output_file}")
    plt.show()
    
    # --- Plot 5: L2 Error (Relative) ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(noise_levels, l2_error_relative, label='L2 Error (Relative)', marker='d', markersize=5, linewidth=2, color='#d62728')
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('L2 Error (Relative)', fontsize=12)
    ax.set_title('Relative L2 Error vs Noise Level', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'verification_l2_error_relative{output_suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Relative L2 Error plot saved to: {output_file}")
    plt.show()
    
    # --- Plot 6: L∞ Error ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(noise_levels, linf_error, label='L∞ Error', marker='v', markersize=5, linewidth=2, color='#9467bd')
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('L∞ Error', fontsize=12)
    ax.set_title('L∞ (Supremum Norm) Error vs Noise Level', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'verification_linf_error{output_suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"L∞ Error plot saved to: {output_file}")
    plt.show()
    
    # --- Plot 7: L∞ Error (Relative) ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(noise_levels, linf_error_relative, label='L∞ Error (Relative)', marker='p', markersize=5, linewidth=2, color='#8c564b')
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('L∞ Error (Relative)', fontsize=12)
    ax.set_title('Relative L∞ (Supremum Norm) Error vs Noise Level', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'verification_linf_error_relative{output_suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Relative L∞ Error plot saved to: {output_file}")
    plt.show()


def main():
    """Main function with command-line argument support."""
    
    parser = argparse.ArgumentParser(description='Plot verification metrics vs noise level from CSV')
    
    parser.add_argument('csv_file', type=str,
                        help='Path to verification_results.csv file')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: same as CSV file directory)')
    
    parser.add_argument('--log_scale', action='store_true',
                        help='Use logarithmic scale for y-axis')
    
    args = parser.parse_args()
    
    # Validate CSV file exists
    if not os.path.isfile(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)
    
    # Plot verification metrics
    plot_verification_metrics(args.csv_file, args.output_dir, args.log_scale)


if __name__ == '__main__':
    main()
