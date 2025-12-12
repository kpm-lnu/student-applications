#!/usr/bin/env python3
"""
Generate large-scale dataset of harmonic functions with boundary conditions.

This script is aimed to generate 100k-200k samples of harmonic functions with:
- Interior solution u(x,y) on the geometry
- Dirichlet boundary conditions u(x,y) on Γ1
- Neumann boundary conditions ∂u/∂n(x,y) on Γ1

The output is saved as 'harmonic_field_dataset.pt' for training the diffusion model.
"""

import numpy as np
import torch
from torch.utils.data import random_split
from tqdm import tqdm
import argparse
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harmonic.sample_harmonic_field import sample_harmonic, compute_normal_derivative
from grid.compute_grid import compute_geometry_mask, compute_boundary_mask
from dataset.compute_stats import compute_global_stats


def setup_geometry(pixel_res=64):
    """
    Hardcode geometry setup for dataset generation.
    Returns geometry_mask, boundary_mask, x_grid, y_grid, and boundary coordinates

    Parameters:
    - pixel_res: grid resolution
    """
    # Parameterize boundary curves Γ1, Γ2
    M = 128
    t = np.linspace(0, 2*np.pi, M, endpoint=True)

    x_Γ1 = 1.3 * np.cos(t)
    y_Γ1 = np.sin(t)

    x_Γ2 = 0.5 * np.cos(t)
    y_Γ2 = 0.4 * np.sin(t) - 0.3 * np.sin(t)**2

    square_bounds = (-1.5, 1.5, -1.5, 1.5)

    # Compute the geometry mask
    geometry_mask, x_grid, y_grid = compute_geometry_mask(
        outer_curve_x=x_Γ1,
        outer_curve_y=y_Γ1,
        inner_curves_x=x_Γ2,
        inner_curves_y=y_Γ2,
        square_bounds=square_bounds,
        pixel_res=pixel_res,
        exclude_boundary=True
    )

    # Compute the boundary mask
    boundary_mask, _, _ = compute_boundary_mask(
        outer_curve_x=x_Γ1,
        outer_curve_y=y_Γ1,
        square_bounds=square_bounds,
        pixel_res=pixel_res,
    )

    # Get boundary point coordinates
    boundary_indices = np.where(boundary_mask == 1)
    boundary_x = x_grid[boundary_indices]
    boundary_y = y_grid[boundary_indices]

    # Get interior point coordinates
    interior_indices = np.where(geometry_mask == 1)
    interior_x = x_grid[interior_indices]
    interior_y = y_grid[interior_indices]

    return {
        'geometry_mask': geometry_mask,
        'boundary_mask': boundary_mask,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'boundary_indices': boundary_indices,
        'boundary_x': boundary_x,
        'boundary_y': boundary_y,
        'interior_indices': interior_indices,
        'interior_x': interior_x,
        'interior_y': interior_y
    }


def generate_diverse_coefficients(N, diversity_mode='mixed'):
    """
    Generate diverse coefficient arrays to create varied boundary conditions.
    
    Parameters:
    - N: number of coefficients for harmonic basis functions
    - diversity_mode: sampling strategy
        - 'mixed': random choice of strategies
        - 'sparse': few dominant terms (focused solutions)
        - 'concentrated': power-law distributed (realistic)
        - 'scaled': variable magnitude scaling
        - 'oscillatory': alternating signs with decay
        - 'clustered': coefficients in frequency bands
    
    Returns:
    - a: coefficient array of shape (N,)
    """
    if diversity_mode == 'mixed':
        # Randomly choose strategy for maximum diversity
        strategies = ['sparse', 'concentrated', 'scaled', 'oscillatory', 'clustered']
        diversity_mode = np.random.choice(strategies)
    
    if diversity_mode == 'sparse':
        # Sparse: only few terms dominate (creates focused solutions)
        a = np.zeros(N)
        num_active = np.random.randint(1, min(8, N//2 + 1))  # 1-8 active terms
        active_indices = np.random.choice(N, num_active, replace=False)
        # Use exponential distribution for magnitudes to get variety
        magnitudes = np.random.exponential(2.0, num_active)
        signs = np.random.choice([-1, 1], num_active)
        a[active_indices] = signs * magnitudes
        
    elif diversity_mode == 'concentrated':
        # Power-law distribution: realistic frequency content
        # Higher frequencies (larger n) have smaller coefficients on average
        n_indices = np.arange(1, N+1)
        # Power law: |a_n| ~ n^(-alpha), alpha ∈ [0.5, 2.5]
        alpha = np.random.uniform(0.5, 2.5)
        base_magnitudes = n_indices**(-alpha)
        # Add random scaling and signs
        random_factors = np.random.exponential(1.0, N)
        signs = np.random.choice([-1, 1], N)
        a = signs * base_magnitudes * random_factors
        
    elif diversity_mode == 'scaled':
        # Variable magnitude scaling: some samples with large/small overall scale
        scale = np.random.choice([
            np.random.uniform(0.1, 0.5),   # Small amplitude solutions
            np.random.uniform(0.5, 2.0),   # Medium amplitude solutions  
            np.random.uniform(2.0, 5.0),   # Large amplitude solutions
            np.random.uniform(5.0, 10.0)   # Very large amplitude solutions
        ])
        a = np.random.randn(N) * scale
        
    elif diversity_mode == 'oscillatory':
        # Alternating signs with decay - creates wave-like patterns
        decay_rate = np.random.uniform(0.8, 1.2)  # Decay factor
        n_indices = np.arange(N)
        alternating = (-1)**n_indices
        decay = decay_rate**(-n_indices)
        base_amplitude = np.random.uniform(0.5, 3.0)
        a = alternating * decay * base_amplitude
        # Add some randomness
        a += np.random.normal(0, 0.1, N)
        
    elif diversity_mode == 'clustered':
        # Concentrated in frequency bands
        a = np.zeros(N)
        # Choose 1-3 frequency bands
        num_bands = np.random.randint(1, 4)
        band_size = max(2, N // 6)  # Band width
        
        for _ in range(num_bands):
            # Random band center
            center = np.random.randint(0, max(1, N - band_size))
            end = min(N, center + band_size)
            
            # Fill band with correlated coefficients
            band_amplitude = np.random.uniform(0.5, 3.0)
            band_coeffs = np.random.randn(end - center) * band_amplitude
            a[center:end] += band_coeffs
    
    else:
        # Fallback: standard normal
        a = np.random.randn(N)
    
    # Ensure we don't get degenerate (all-zero) solutions
    if np.allclose(a, 0):
        a = np.random.randn(N) * 0.1
    
    return a


def generate_harmonic_sample(geometry_info, N=8, random_seed=None, diversity_mode='mixed'):
    """
    Generate a single harmonic function sample with boundary conditions.
    
    Parameters:
    - geometry_info: dictionary with geometry setup from setup_geometry()
    - N: number of harmonic basis functions
    - random_seed: optional seed for reproducibility
    - diversity_mode: strategy for coefficient sampling ('mixed', 'sparse', 'concentrated', 'scaled')
    
    Returns:
    - sample_dict: dictionary with keys 'u', 'gmask', 'bmask', 'dirichlet', 'neumann'
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate diverse random coefficients using different strategies
    a = generate_diverse_coefficients(N, diversity_mode)

    # Unpack geometry info    
    geometry_mask = geometry_info['geometry_mask']
    boundary_mask = geometry_info['boundary_mask']
    interior_indices = geometry_info['interior_indices']
    boundary_indices = geometry_info['boundary_indices']
    interior_x = geometry_info['interior_x']
    interior_y = geometry_info['interior_y']
    boundary_x = geometry_info['boundary_x']
    boundary_y = geometry_info['boundary_y']
    
    # Sample harmonic field on boundary points
    u_boundary = sample_harmonic(boundary_x, boundary_y, a)
    du_dn_boundary = compute_normal_derivative(boundary_x, boundary_y, a)

    # Filter out invalid values (NaN or Inf)
    valid_mask = np.isfinite(u_boundary) & np.isfinite(du_dn_boundary)
    
    if not np.all(valid_mask):
        # Replace invalid values with zeros
        u_boundary[~valid_mask] = 0.0
        du_dn_boundary[~valid_mask] = 0.0
    
    # Create Dirichlet and Neumann masks
    boundary_dirichlet_mask = np.zeros_like(boundary_mask, dtype=np.float32)
    boundary_dirichlet_mask[boundary_indices] = u_boundary

    boundary_neumann_mask = np.zeros_like(boundary_mask, dtype=np.float32)
    boundary_neumann_mask[boundary_indices] = du_dn_boundary

    # Sample harmonic field on interior points using same coefficients
    u_interior = sample_harmonic(interior_x, interior_y, a)
    
    # Create full solution mask (interior + boundary values)
    u_full = np.zeros_like(geometry_mask, dtype=np.float32)
    u_full[interior_indices] = u_interior
    u_full[boundary_indices] = u_boundary  # Include boundary values in solution
    
    return {
        'u': u_full,
        'gmask': geometry_mask.astype(np.float32),
        'bmask': boundary_mask.astype(np.float32),
        'dirichlet': boundary_dirichlet_mask,
        'neumann': boundary_neumann_mask
    }


def validate_sample(sample_dict, geometry_info):
    """
    Validate a generated sample for basic sanity checks.
    
    Returns:
    - is_valid: bool indicating if sample passes validation
    - validation_info: dict with validation metrics
    """
    u = sample_dict['u']
    dirichlet = sample_dict['dirichlet']
    neumann = sample_dict['neumann']
    
    # Check for finite values
    u_finite = np.isfinite(u).all()
    d_finite = np.isfinite(dirichlet).all()
    n_finite = np.isfinite(neumann).all()
    
    # Check that solution has non-trivial values
    interior_points = geometry_info['geometry_mask'].sum()
    boundary_points = geometry_info['boundary_mask'].sum()
    solution_points = (u != 0).sum()
    
    validation_info = {
        'u_finite': u_finite,
        'dirichlet_finite': d_finite,
        'neumann_finite': n_finite,
        'interior_coverage': solution_points / max(interior_points + boundary_points, 1),
        'u_range': (float(u.min()), float(u.max())),
        'dirichlet_range': (float(dirichlet.min()), float(dirichlet.max())),
        'neumann_range': (float(neumann.min()), float(neumann.max()))
    }
    
    # Sample is valid if all values are finite and solution is non-trivial
    is_valid = (u_finite and d_finite and n_finite and 
                validation_info['interior_coverage'] > 0.9)
    
    return is_valid, validation_info


def generate_dataset(output_file="harmonic_field_dataset.pt",
                    num_samples=1000,
                    pixel_res=64,
                    N=8,
                    seed=42,
                    val_split=0.2):
    """
    Generate a large dataset of harmonic function samples.
    
    Parameters:
    - output_file: output filename
    - num_samples: number of samples to generate
    - pixel_res: grid resolution (pixel_res x pixel_res)
    - N: number of harmonic basis functions
    - seed: random seed for reproducibility
    - val_split: fraction of data to use for validation (default: 0.2)
    """
    print(f"Generating dataset with {num_samples} samples...")
    print(f"Grid pixel resolution: {pixel_res}x{pixel_res}")
    print(f"Harmonic basis functions: {N}")
    print(f"Output file: {output_file}")
    
    # Set global random seed
    np.random.seed(seed)
    
    # Setup geometry
    print("Setting up geometry...")
    geometry_info = setup_geometry(pixel_res)
    
    print(f"Geometry info:")
    print(f"  Interior points: {geometry_info['geometry_mask'].sum()}")
    print(f"  Boundary points: {geometry_info['boundary_mask'].sum()}")
    print(f"  Total grid points: {pixel_res}x{pixel_res} = {pixel_res**2}")
    
    # Generate samples
    all_samples = []
    
    validation_stats = {
        'valid_samples': 0,
        'invalid_samples': 0,
        'validation_errors': []
    }
    
    print(f"\nGenerating {num_samples} samples...")
    
    for i in tqdm(range(num_samples), desc="Generating samples"):
        # Use different seed for each sample
        sample_seed = seed + i
        
        try:
            is_valid_sample = False
            while not is_valid_sample:
                # Use mixed diversity mode for maximum variety
                sample = generate_harmonic_sample(geometry_info, N=N, random_seed=sample_seed, diversity_mode='mixed')
                
                # Validate sample
                is_valid, val_info = validate_sample(sample, geometry_info)
                
                if is_valid:
                    all_samples.append(sample)
                    validation_stats['valid_samples'] += 1
                    is_valid_sample = True
                else:
                    validation_stats['invalid_samples'] += 1
                    validation_stats['validation_errors'].append(val_info)
                
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            validation_stats['invalid_samples'] += 1
        
    print(f"\nGenerated {len(all_samples)} samples")
    print(f"Valid samples: {validation_stats['valid_samples']}")
    print(f"Invalid samples: {validation_stats['invalid_samples']}")
    
    # Split dataset into train and validation
    print(f"\nSplitting dataset: {100*(1-val_split):.0f}% train, {100*val_split:.0f}% validation")
    total_samples = len(all_samples)
    train_size = int((1 - val_split) * total_samples)
    val_size = total_samples - train_size
    
    # Create split using the same seed
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(total_samples),
        [train_size, val_size],
        generator=generator
    )
    
    train_samples = [all_samples[i] for i in train_indices.indices]
    val_samples = [all_samples[i] for i in val_indices.indices]
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    
    # Compute global statistics from TRAINING SET ONLY
    print("\nComputing normalization statistics from training set only...")
    mu, sigma = compute_global_stats(train_samples)
    print(f"Global statistics (from train): μ = {mu:.6f}, σ = {sigma:.6f}")
    
    # Save train and validation datasets
    base_name = output_file.replace('.pt', '')
    train_file = f"{base_name}_train.pt"
    val_file = f"{base_name}_val.pt"
    
    print(f"\nSaving training dataset to {train_file}...")
    torch.save(train_samples, train_file)
    
    print(f"Saving validation dataset to {val_file}...")
    torch.save(val_samples, val_file)
    
    # Save metadata (same for both train and val)
    metadata = {
        'num_samples': total_samples,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'val_split': val_split,
        'pixel_res': pixel_res,
        'N': N,
        'global_mu': mu,
        'global_sigma': sigma,
        'validation_stats': validation_stats,
        'geometry_info': {
            'interior_points': int(geometry_info['geometry_mask'].sum()),
            'boundary_points': int(geometry_info['boundary_mask'].sum()),
        }
    }
    
    metadata_file = f"{base_name}_metadata.pt"
    torch.save(metadata, metadata_file)
    print(f"Saved metadata to {metadata_file}")
    
    print("\nDataset generation completed successfully!")
    print(f"Files created:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print(f"  - {metadata_file}")
    
    return train_samples, val_samples, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate harmonic function dataset")
    parser.add_argument("--num_samples", type=int, default=1000, 
                       help="Number of samples to generate (default: 1000)")
    parser.add_argument("--pixel_res", type=int, default=64,
                       help="Grid resolution (default: 64)")
    parser.add_argument("--N", type=int, default=8,
                       help="Number of harmonic basis functions (default: 8)")
    parser.add_argument("--output", type=str, default="harmonic_field_dataset.pt",
                       help="Output filename (default: harmonic_field_dataset.pt)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--val_split", type=float, default=0.2,
                       help="Fraction of data for validation (default: 0.2)")
    
    args = parser.parse_args()
    
    # Generate dataset
    train_samples, val_samples, metadata = generate_dataset(
        num_samples=args.num_samples,
        pixel_res=args.pixel_res,
        N=args.N,
        output_file=args.output,
        seed=args.seed,
        val_split=args.val_split
    )
    
    base_name = args.output.replace('.pt', '')
    print(f"\nGenerated {len(train_samples) + len(val_samples)} samples successfully!")
    print(f"Training dataset saved to: {base_name}_train.pt ({len(train_samples)} samples)")
    print(f"Validation dataset saved to: {base_name}_val.pt ({len(val_samples)} samples)")
    print(f"Metadata saved to: {base_name}_metadata.pt")
    print(f"\nMetadata summary:")
    print(f"  Total samples: {metadata['num_samples']}")
    print(f"  Train samples: {metadata['train_samples']}")
    print(f"  Val samples: {metadata['val_samples']}")
    print(f"  Normalization: μ={metadata['global_mu']:.6f}, σ={metadata['global_sigma']:.6f}")


if __name__ == "__main__":
    main()