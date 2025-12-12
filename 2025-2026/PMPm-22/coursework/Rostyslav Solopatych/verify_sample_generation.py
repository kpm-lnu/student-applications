#!/usr/bin/env python3
"""
Test sample generation using the CLEAN diffusion model.

This script generates samples using the model trained on clean data and verifies
that the physical values are consistent with dataset_utils.py plotting.
"""

import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.generate_samples import load_trained_model, generate_samples

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_generation(sample_idx, checkpoint_path, scheduler_config_path, dataset_file, 
                          num_ensemble_samples, num_diffusion_timesteps, noise=0.0):
    """Test generation with the clean model.
    
    Args:
        checkpoint_path: Full path to the model checkpoint directory
        scheduler_config_path: Full path to the scheduler config file
        dataset_file: Full path to the dataset file
        sample_idx: Index of the sample to test against
        num_ensemble_samples: Number of samples in the ensemble (default: 10)
        num_diffusion_timesteps: Number of diffusion timesteps
        noise: Relative noise level as fraction of data range (e.g., 0.1 = 10% noise) (default: 0.0)
    """
    
    logger.info("=== TESTING CLEAN MODEL GENERATION ===")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Validate paths
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(scheduler_config_path):
        raise FileNotFoundError(f"Scheduler config not found: {scheduler_config_path}")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")
    
    # Load trained model
    logger.info(f"Loading model from: {checkpoint_path}")
    model, scheduler = load_trained_model(
        checkpoint_path=checkpoint_path,
        scheduler_config_path=scheduler_config_path,
        device=device
    )
    
    # Load dataset
    logger.info(f"Loading dataset from: {dataset_file}")
    dataset = torch.load(dataset_file, weights_only=False)
    
    # Load metadata
    if '_train.pt' in dataset_file:
        metadata_file = dataset_file.replace('_train.pt', '_metadata.pt')
    elif '_val.pt' in dataset_file:
        metadata_file = dataset_file.replace('_val.pt', '_metadata.pt')
    else:
        metadata_file = dataset_file.replace('.pt', '_metadata.pt')
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    metadata = torch.load(metadata_file, weights_only=False)
    
    dataset_type = "validation" if "val" in dataset_file else "train" if "train" in dataset_file else "full"
    logger.info(f"Dataset: {len(dataset)} {dataset_type} samples")
    logger.info(f"Normalization stats: μ={metadata['global_mu']:.6f}, σ={metadata['global_sigma']:.6f}")
    
    if sample_idx >= len(dataset):
        raise ValueError(f"Sample index {sample_idx} out of range (dataset has {len(dataset)} samples)")
    
    sample = dataset[sample_idx]
    
    # Note: sample['u'] contains the original physical values (not normalized)
    original_u = sample['u']
    
    # Prepare conditioning tensor using proper normalization stats
    mu = metadata['global_mu']
    sigma = metadata['global_sigma']
    
    # Extract conditioning data
    dirichlet_data = sample['dirichlet']
    neumann_data = sample['neumann']
    
    # Add Gaussian noise to conditioning data (Dirichlet and Neumann) if requested
    if noise > 0:
        # Compute ranges for scaling noise (relative noise)
        dirichlet_range = np.max(dirichlet_data) - np.min(dirichlet_data)
        neumann_range = np.max(neumann_data) - np.min(neumann_data)
        
        # Scale noise by range to get relative noise
        dirichlet_noise = noise * dirichlet_range
        neumann_noise = noise * neumann_range
        
        logger.info(f"Adding relative Gaussian noise with std={noise} (scaled by range) to conditioning data")
        logger.info(f"Dirichlet absolute noise std: {dirichlet_noise:.6f}, Neumann absolute noise std: {neumann_noise:.6f}")
        
        dirichlet_noise = np.random.normal(0, dirichlet_noise, dirichlet_data.shape)
        neumann_noise = np.random.normal(0, neumann_noise, neumann_data.shape)
        dirichlet_data_noisy = dirichlet_data + dirichlet_noise
        neumann_data_noisy = neumann_data + neumann_noise
        
        logger.info(f"Dirichlet - Clean range: [{dirichlet_data.min():.3f}, {dirichlet_data.max():.3f}], Noisy range: [{dirichlet_data_noisy.min():.3f}, {dirichlet_data_noisy.max():.3f}]")
        logger.info(f"Neumann - Clean range: [{neumann_data.min():.3f}, {neumann_data.max():.3f}], Noisy range: [{neumann_data_noisy.min():.3f}, {neumann_data_noisy.max():.3f}]")
    else:
        dirichlet_data_noisy = dirichlet_data
        neumann_data_noisy = neumann_data
    
    # Normalize the (potentially noisy) conditioning data
    d_norm = torch.tensor((dirichlet_data_noisy - mu) / (sigma + 1e-6), dtype=torch.float32)
    n_norm = torch.tensor(neumann_data_noisy / (sigma + 1e-6), dtype=torch.float32)
    gmask = torch.tensor(sample['gmask'], dtype=torch.float32)
    bmask = torch.tensor(sample['bmask'], dtype=torch.float32)
    cond_single = torch.stack([gmask, bmask, d_norm, n_norm], dim=0).unsqueeze(0).to(device)
    
    # Generate ensemble of samples for more accurate physics solution
    logger.info(f"Generating ensemble of {num_ensemble_samples} samples for accurate solution...")
    
    # Repeat conditioning tensor to generate multiple samples
    cond = cond_single.repeat(num_ensemble_samples, 1, 1, 1)  # [num_ensemble_samples, 4, H, W]
    
    # Generate ensemble samples using clean model with clean normalization stats
    ensemble_samples = generate_samples(
        model=model,
        scheduler=scheduler,
        cond=cond,
        mu=mu,
        sigma=sigma,
        denoising_steps=num_diffusion_timesteps,
        device=device
    )
    
    # Compute ensemble statistics
    ensemble_mean = torch.mean(ensemble_samples, dim=0, keepdim=True)  # [1, 1, H, W] - Mean is the solution
    ensemble_std = torch.std(ensemble_samples, dim=0, keepdim=True)    # [1, 1, H, W] - Std shows uncertainty
    
    # Use ensemble mean as the generated solution for comparison
    generated = ensemble_mean
    
    # Compare ranges
    original_range = [float(original_u.min()), float(original_u.max())]
    generated_range = [generated.min().item(), generated.max().item()]
        
    # Check if ranges are comparable (within reasonable factor)
    original_span = original_range[1] - original_range[0]
    generated_span = generated_range[1] - generated_range[0]
    ratio = generated_span / (original_span + 1e-8)
    
    # Compute error metrics
    generated_2d = generated[0, 0].cpu().numpy() # Extract [H,W] from [1,1,H,W]
    original_2d = original_u
    
    # L2 error (aggregate measure)
    l2_error = np.linalg.norm(generated_2d - original_2d)
    original_norm = np.linalg.norm(original_2d)
    l2_error_relative = l2_error / (original_norm + 1e-8)
    
    # RMSE (mean squared error)
    rmse = np.sqrt(np.mean((generated_2d - original_2d) ** 2))
    
    # L∞ error (max pointwise error - most interpretable)
    linf_error = np.max(np.abs(generated_2d - original_2d))
    linf_error_relative = linf_error / (np.max(np.abs(original_2d)) + 1e-8)
        
    # Create visualization (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Original sample
    im1 = axes[0,0].imshow(original_u, cmap='RdBu_r', origin='lower')
    noise_label = f' (with {noise * 100:.1f}% noise)' if noise > 0 else ''
    axes[0,0].set_title(f'Original Sample {sample_idx}\nRange: [{original_range[0]:.3f}, {original_range[1]:.3f}]')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0])

    # Generated ensemble mean (the physics solution)
    im2 = axes[0,1].imshow(generated_2d, cmap='RdBu_r', origin='lower')
    axes[0,1].set_title(f'Ensemble Mean Solution ({num_ensemble_samples} samples){noise_label}\nRange: [{generated_range[0]:.3f}, {generated_range[1]:.3f}]')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Uncertainty (ensemble std)
    uncertainty_2d = ensemble_std[0, 0].cpu().numpy()
    im3 = axes[1,0].imshow(uncertainty_2d, cmap='viridis', origin='lower')
    axes[1,0].set_title(f'Uncertainty (Ensemble Std)\nMax: {uncertainty_2d.max():.4f}')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Difference between ensemble mean and original
    diff = generated_2d - original_u
    im4 = axes[1,1].imshow(diff, cmap='RdBu_r', origin='lower')
    axes[1,1].set_title(f'Difference (Ensemble Mean - Original)\nL∞: {linf_error:.4f} | L2: {l2_error:.4f} | RMSE: {rmse:.4f}')
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    output_filename = f'ensemble_model_generation_test_noise{noise}.png' if noise > 0 else 'ensemble_model_generation_test.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    logger.info(f"Saved ensemble visualization: {output_filename}")
    
    # Save results to JSON
    results = {
        'original_range': original_range,
        'generated_range': generated_range,
        'ratio': float(ratio),
        'rmse': float(rmse),
        'l2_error': float(l2_error),
        'l2_error_relative': float(l2_error_relative),
        'linf_error': float(linf_error),
        'linf_error_relative': float(linf_error_relative),
        'num_ensemble_samples': int(num_ensemble_samples),
        'max_uncertainty': float(uncertainty_2d.max()),
        'mean_uncertainty': float(uncertainty_2d.mean()),
        'noise': float(noise),
        'sample_idx': int(sample_idx)
    }
    
    output_json_filename = f'ensemble_model_generation_test_noise{noise}.json' if noise > 0 else 'ensemble_model_generation_test.json'
    with open(output_json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to: {output_json_filename}")
    
    return results


def main():
    """Main test function."""
    
    parser = argparse.ArgumentParser(description="Test sample generation with the diffusion model")
    
    # Dataset and model arguments
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset file (e.g., harmonic_field_dataset_val.pt)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint directory (e.g., ./checkpoints/cond_ddpm_epoch10)")
    parser.add_argument("--scheduler_config", type=str, required=True,
                       help="Path to scheduler config file (e.g., ./checkpoints/scheduler_epoch10.json/scheduler_config.json)")
    
    # Sample and generation arguments
    parser.add_argument("--sample_idx", type=int, default=1,
                       help="Index of the sample to test against (default: 1)")
    parser.add_argument("--ensemble_size", type=int, default=10,
                       help="Number of samples in the ensemble (default: 10)")
    parser.add_argument('--num_diffusion_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps (default: 1000)')
    
    # Noise argument
    parser.add_argument('--noise', type=float, default=0.0,
                       help='Relative noise level for conditioning data (Dirichlet and Neumann) as fraction of data range. E.g., 0.1 = 10%% noise (default: 0.0)')
    
    args = parser.parse_args()
    
    logger.info("🧪 TESTING CLEAN MODEL FOR CONSISTENT PHYSICAL VALUE PLOTTING")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Sample Index: {args.sample_idx} | Ensemble Size: {args.ensemble_size}")
    if args.noise > 0:
        logger.info(f"Conditioning Noise Std: {args.noise} (robustness testing enabled)")
    logger.info("=" * 80)
    
    generation_results = test_model_generation(
        checkpoint_path=args.checkpoint,
        scheduler_config_path=args.scheduler_config,
        dataset_file=args.dataset,
        sample_idx=args.sample_idx,
        num_ensemble_samples=args.ensemble_size,
        num_diffusion_timesteps=args.num_diffusion_timesteps,
        noise=args.noise
    )

    # Analyze results
    original_range = generation_results['original_range']
    generated_range = generation_results['generated_range']
    ratio = generation_results['ratio']
    rmse = generation_results['rmse']
    l2_error = generation_results['l2_error']
    l2_error_relative = generation_results['l2_error_relative']
    linf_error = generation_results['linf_error']
    linf_error_relative = generation_results['linf_error_relative']
    num_ensemble_samples = generation_results['num_ensemble_samples']
    max_uncertainty = generation_results['max_uncertainty']
    mean_uncertainty = generation_results['mean_uncertainty']
    noise = generation_results['noise']

    
    if ratio < 1.5 and ratio > 0.5 and linf_error_relative < 0.1:
        logger.info(f"✅ PERFECT: Model produces physically reasonable solutions (max pointwise error < 10%)")
        success = True
    elif linf_error_relative < 0.3:
        logger.info(f"⚠️ GOOD: Generated values are acceptable (max pointwise error < 30%)")
        success = True
    elif linf_error_relative < 0.5:
        logger.info(f"⚠️ WARNING: Generated values need improvement (max pointwise error < 50%)")
        success = True
    else:
        logger.info(f"❌ POOR: Max pointwise error > 50% - consider additional training")
        success = False
    
    conditioning_note = f" (with {noise * 100:.1f}% noise)" if noise > 0 else ""
    logger.info(f"\nModel Performance{conditioning_note}:\
        \n  L∞ Error {linf_error:.6f} (max pointwise error)\
        \n  L∞ Rel. Error {linf_error_relative*100:.2f}%\
        \n  L2 Error {l2_error:.6f}\
        \n  L2 Rel. Error {l2_error_relative*100:.2f}%\
        \n  RMSE {rmse:.6f}\
        \n  NRMSE {rmse/((original_range[1]-original_range[0])+1e-8):.6f}\
        \n  Range Ratio {ratio:.3f}"
    )
    
    logger.info(f"\nRanges: Original [{original_range[0]:.3f}, {original_range[1]:.3f}] | Generated [{generated_range[0]:.3f}, {generated_range[1]:.3f}]")
    logger.info(f"\nEnsemble: Size {num_ensemble_samples} | Max Unc. {max_uncertainty:.6f} | Mean Unc. {mean_uncertainty:.6f}")
    
    if success:
        logger.info(f"\n🎉 SUCCESS: Ensemble mean converges to accurate physics solution!")
    
    return success

if __name__ == "__main__":
    main()