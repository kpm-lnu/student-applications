#!/usr/bin/env python3
"""
Analyze dataset to understand value ranges and normalization impact.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import defaultdict

def analyze_dataset(dataset_path, metadata_path, output_dir):
    """
    Analyze the dataset to understand value distributions and normalization.
    
    Parameters:
    - dataset_path: path to train or val .pt file
    - metadata_path: path to metadata .pt file
    - output_dir: directory where all output files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading dataset from {dataset_path}...")
    data = torch.load(dataset_path, weights_only=False)
    
    print(f"Loading metadata from {metadata_path}...")
    metadata = torch.load(metadata_path, weights_only=False)
    
    mu = metadata['global_mu']
    sigma = metadata['global_sigma']
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Number of samples: {len(data)}")
    print(f"Normalization: μ={mu:.6f}, σ={sigma:.6f}")
    
    # Collect statistics for each sample
    sample_stats = []
    raw_values = []
    normalized_values = []
    
    for i, sample in enumerate(data):
        u = sample['u']
        u_min = float(u.min())
        u_max = float(u.max())
        u_range = u_max - u_min
        
        # Compute normalized values
        u_norm = (u - mu) / sigma
        norm_min = float(u_norm.min())
        norm_max = float(u_norm.max())
        
        sample_stats.append({
            'idx': i,
            'raw_min': u_min,
            'raw_max': u_max,
            'raw_range': u_range,
            'norm_min': norm_min,
            'norm_max': norm_max,
            'norm_range': norm_max - norm_min
        })
        
        # Collect all values for distribution
        raw_values.extend(u.flatten().tolist())
        normalized_values.extend(u_norm.flatten().tolist())
    
    raw_values = np.array(raw_values)
    normalized_values = np.array(normalized_values)
    
    # Convert to numpy array for easier analysis
    ranges = np.array([s['raw_range'] for s in sample_stats])
    norm_mins = np.array([s['norm_min'] for s in sample_stats])
    norm_maxs = np.array([s['norm_max'] for s in sample_stats])
    
    print(f"\n=== Raw Value Statistics ===")
    print(f"Overall min: {raw_values.min():.2f}")
    print(f"Overall max: {raw_values.max():.2f}")
    print(f"Overall range: {raw_values.max() - raw_values.min():.2f}")
    print(f"Mean: {raw_values.mean():.6f}")
    print(f"Std: {raw_values.std():.6f}")
    
    print(f"\n=== Per-Sample Range Statistics ===")
    print(f"Min range: {ranges.min():.2f}")
    print(f"Max range: {ranges.max():.2f}")
    print(f"Mean range: {ranges.mean():.2f}")
    print(f"Median range: {np.median(ranges):.2f}")
    print(f"25th percentile: {np.percentile(ranges, 25):.2f}")
    print(f"75th percentile: {np.percentile(ranges, 75):.2f}")
    print(f"95th percentile: {np.percentile(ranges, 95):.2f}")
    print(f"99th percentile: {np.percentile(ranges, 99):.2f}")
    
    print(f"\n=== Normalized Value Statistics ===")
    print(f"Normalized min: {normalized_values.min():.2f}σ")
    print(f"Normalized max: {normalized_values.max():.2f}σ")
    print(f"Normalized range: {normalized_values.max() - normalized_values.min():.2f}σ")
    print(f"% values outside [-3σ, 3σ]: {(np.abs(normalized_values) > 3).sum() / len(normalized_values) * 100:.2f}%")
    print(f"% values outside [-5σ, 5σ]: {(np.abs(normalized_values) > 5).sum() / len(normalized_values) * 100:.2f}%")
    print(f"% values outside [-10σ, 10σ]: {(np.abs(normalized_values) > 10).sum() / len(normalized_values) * 100:.2f}%")
    
    # Find samples with extreme normalized values
    print(f"\n=== Samples with Extreme Normalized Values ===")
    extreme_samples = [s for s in sample_stats if abs(s['norm_min']) > 5 or abs(s['norm_max']) > 5]
    print(f"Number of samples with |norm_value| > 5σ: {len(extreme_samples)}")
    
    if extreme_samples:
        print(f"\nTop 10 most extreme samples:")
        extreme_samples.sort(key=lambda x: max(abs(x['norm_min']), abs(x['norm_max'])), reverse=True)
        for s in extreme_samples[:10]:
            print(f"  Sample {s['idx']}: raw=[{s['raw_min']:.2f}, {s['raw_max']:.2f}], "
                  f"norm=[{s['norm_min']:.2f}σ, {s['norm_max']:.2f}σ]")
    
    # Check outlier thresholds
    print(f"\n=== Outlier Threshold Analysis ===")
    for percentile in [0.1, 0.5, 1.0, 2.0, 5.0]:
        low = np.percentile(raw_values, percentile)
        high = np.percentile(raw_values, 100 - percentile)
        values_in_range = ((raw_values >= low) & (raw_values <= high)).sum()
        pct_included = values_in_range / len(raw_values) * 100
        print(f"Percentile {percentile}%: [{low:.2f}, {high:.2f}] includes {pct_included:.2f}% of values")
    
    # Visualizations
    print(f"\n=== Generating Plots ===")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Raw value distribution
    axes[0, 0].hist(raw_values, bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(mu, color='r', linestyle='--', label=f'μ={mu:.2f}')
    axes[0, 0].axvline(mu - sigma, color='orange', linestyle='--', label=f'±σ')
    axes[0, 0].axvline(mu + sigma, color='orange', linestyle='--')
    axes[0, 0].set_xlabel('Raw value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Raw Value Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Normalized value distribution
    axes[0, 1].hist(normalized_values, bins=100, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(0, color='r', linestyle='--', label='μ=0')
    axes[0, 1].axvline(-3, color='orange', linestyle='--', label='±3σ')
    axes[0, 1].axvline(3, color='orange', linestyle='--')
    axes[0, 1].set_xlabel('Normalized value (σ)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Normalized Value Distribution')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Per-sample range distribution
    axes[0, 2].hist(ranges, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 2].axvline(np.median(ranges), color='r', linestyle='--', label=f'Median={np.median(ranges):.2f}')
    axes[0, 2].set_xlabel('Sample range')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Per-Sample Range Distribution')
    axes[0, 2].legend()
    
    # Plot 4: Normalized min/max scatter
    axes[1, 0].scatter(norm_mins, norm_maxs, alpha=0.5, s=10)
    axes[1, 0].axhline(3, color='orange', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(-3, color='orange', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(3, color='orange', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(-3, color='orange', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Normalized min (σ)')
    axes[1, 0].set_ylabel('Normalized max (σ)')
    axes[1, 0].set_title('Normalized Range per Sample')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: CDF of normalized absolute values
    abs_norm = np.abs(normalized_values)
    abs_norm_sorted = np.sort(abs_norm)
    cdf = np.arange(1, len(abs_norm_sorted) + 1) / len(abs_norm_sorted)
    axes[1, 1].plot(abs_norm_sorted, cdf * 100)
    axes[1, 1].axvline(3, color='orange', linestyle='--', label='3σ', alpha=0.7)
    axes[1, 1].axvline(5, color='red', linestyle='--', label='5σ', alpha=0.7)
    axes[1, 1].axvline(10, color='darkred', linestyle='--', label='10σ', alpha=0.7)
    axes[1, 1].set_xlabel('|Normalized value| (σ)')
    axes[1, 1].set_ylabel('Cumulative %')
    axes[1, 1].set_title('CDF of Absolute Normalized Values')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Range vs index (to see if there's any pattern)
    axes[1, 2].scatter(range(len(ranges)), ranges, alpha=0.5, s=10)
    axes[1, 2].set_xlabel('Sample index')
    axes[1, 2].set_ylabel('Sample range')
    axes[1, 2].set_title('Sample Range vs Index')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    output_file = os.path.join(output_dir, f'{base_name}_analysis.png')
    plt.savefig(output_file, dpi=150)
    print(f"Saved plot to {output_file}")
    plt.show()
    
    # Export results to CSV
    print(f"\n=== Exporting Results to CSV ===")
    
    # 1. Per-sample statistics
    df_samples = pd.DataFrame(sample_stats)
    csv_samples = os.path.join(output_dir, f'{base_name}_per_sample_stats.csv')
    df_samples.to_csv(csv_samples, index=False)
    print(f"Saved per-sample statistics to {csv_samples}")
    
    # 2. Overall statistics summary
    summary_data = {
        'metric': [
            'num_samples',
            'normalization_mu',
            'normalization_sigma',
            'raw_min',
            'raw_max',
            'raw_range',
            'raw_mean',
            'raw_std',
            'range_min',
            'range_max',
            'range_mean',
            'range_median',
            'range_25th_percentile',
            'range_75th_percentile',
            'range_95th_percentile',
            'range_99th_percentile',
            'norm_min',
            'norm_max',
            'norm_range',
            'pct_outside_3sigma',
            'pct_outside_5sigma',
            'pct_outside_10sigma',
            'num_extreme_samples_5sigma',
            'outlier_threshold_0.1_low',
            'outlier_threshold_0.1_high',
            'outlier_threshold_1.0_low',
            'outlier_threshold_1.0_high'
        ],
        'value': [
            len(data),
            mu,
            sigma,
            raw_values.min(),
            raw_values.max(),
            raw_values.max() - raw_values.min(),
            raw_values.mean(),
            raw_values.std(),
            ranges.min(),
            ranges.max(),
            ranges.mean(),
            np.median(ranges),
            np.percentile(ranges, 25),
            np.percentile(ranges, 75),
            np.percentile(ranges, 95),
            np.percentile(ranges, 99),
            normalized_values.min(),
            normalized_values.max(),
            normalized_values.max() - normalized_values.min(),
            (np.abs(normalized_values) > 3).sum() / len(normalized_values) * 100,
            (np.abs(normalized_values) > 5).sum() / len(normalized_values) * 100,
            (np.abs(normalized_values) > 10).sum() / len(normalized_values) * 100,
            len(extreme_samples),
            np.percentile(raw_values, 0.1),
            np.percentile(raw_values, 99.9),
            np.percentile(raw_values, 1.0),
            np.percentile(raw_values, 99.0)
        ]
    }
    df_summary = pd.DataFrame(summary_data)
    csv_summary = os.path.join(output_dir, f'{base_name}_summary_stats.csv')
    df_summary.to_csv(csv_summary, index=False)
    print(f"Saved summary statistics to {csv_summary}")
    
    # 3. Value distribution (binned histogram data)
    hist_raw, bin_edges_raw = np.histogram(raw_values, bins=100)
    df_hist_raw = pd.DataFrame({
        'bin_center': (bin_edges_raw[:-1] + bin_edges_raw[1:]) / 2,
        'bin_start': bin_edges_raw[:-1],
        'bin_end': bin_edges_raw[1:],
        'frequency': hist_raw
    })
    csv_hist_raw = os.path.join(output_dir, f'{base_name}_raw_value_histogram.csv')
    df_hist_raw.to_csv(csv_hist_raw, index=False)
    print(f"Saved raw value histogram to {csv_hist_raw}")
    
    hist_norm, bin_edges_norm = np.histogram(normalized_values, bins=100)
    df_hist_norm = pd.DataFrame({
        'bin_center': (bin_edges_norm[:-1] + bin_edges_norm[1:]) / 2,
        'bin_start': bin_edges_norm[:-1],
        'bin_end': bin_edges_norm[1:],
        'frequency': hist_norm
    })
    csv_hist_norm = os.path.join(output_dir, f'{base_name}_normalized_value_histogram.csv')
    df_hist_norm.to_csv(csv_hist_norm, index=False)
    print(f"Saved normalized value histogram to {csv_hist_norm}")
    
    # 4. CDF data
    abs_norm = np.abs(normalized_values)
    abs_norm_sorted = np.sort(abs_norm)
    cdf = np.arange(1, len(abs_norm_sorted) + 1) / len(abs_norm_sorted)
    # Sample every 100th point to reduce file size
    sample_indices = np.linspace(0, len(abs_norm_sorted)-1, min(10000, len(abs_norm_sorted)), dtype=int)
    df_cdf = pd.DataFrame({
        'abs_normalized_value': abs_norm_sorted[sample_indices],
        'cumulative_percentage': cdf[sample_indices] * 100
    })
    csv_cdf = os.path.join(output_dir, f'{base_name}_cdf.csv')
    df_cdf.to_csv(csv_cdf, index=False)
    print(f"Saved CDF data to {csv_cdf}")
    
    print(f"\n✓ All CSV files exported successfully!")
    
    return sample_stats, metadata


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python analyze_dataset.py <dataset.pt> <output_dir> [metadata.pt]")
        print("Example: python analyze_dataset.py harmonic_field_dataset_train.pt analysis_output harmonic_field_dataset_metadata.pt")
        print("  <dataset.pt>  : Path to the dataset file")
        print("  <output_dir>  : Directory where all outputs will be saved")
        print("  [metadata.pt] : Optional path to metadata file (will be inferred if not provided)")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    if len(sys.argv) >= 4:
        metadata_path = sys.argv[3]
    else:
        # Try to infer metadata path
        base = dataset_path.replace('_train.pt', '').replace('_val.pt', '')
        metadata_path = f"{base}_metadata.pt"
    
    analyze_dataset(dataset_path, metadata_path, output_dir)
