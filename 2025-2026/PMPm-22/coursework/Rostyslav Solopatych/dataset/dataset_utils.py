#!/usr/bin/env python3
"""
Utility script for working with generated harmonic datasets.
Provides functions to inspect, validate, and analyze dataset files.
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def inspect_dataset(dataset_file):
    """Inspect a generated dataset file"""
    print(f"Inspecting dataset: {dataset_file}")
    
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset file not found: {dataset_file}")
        return
    
    try:
        # Load dataset with weights_only=False for compatibility
        data = torch.load(dataset_file, map_location='cpu', weights_only=False)
        print(f"✓ Successfully loaded dataset")
        print(f"  Number of samples: {len(data)}")
        
        if len(data) == 0:
            print("❌ Dataset is empty")
            return
        
        # Inspect first sample
        sample = data[0]
        print(f"  Sample keys: {list(sample.keys())}")
        
        for key in sample.keys():
            arr = sample[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            print(f"    range=[{arr.min():.6f}, {arr.max():.6f}], mean={arr.mean():.6f}")
            print(f"    non-zero: {np.count_nonzero(arr)}/{arr.size} ({100*np.count_nonzero(arr)/arr.size:.1f}%)")
        
        # Check metadata if available
        metadata_file = dataset_file.replace('.pt', '_metadata.pt')
        if os.path.exists(metadata_file):
            metadata = torch.load(metadata_file, map_location='cpu', weights_only=False)
            print(f"\n✓ Metadata available:")
            for key, value in metadata.items():
                if key != 'validation_stats':
                    print(f"  {key}: {value}")
                    
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")


def validate_training_compatibility(dataset_file):
    """Test compatibility with training pipeline"""
    print(f"\nValidating training compatibility for: {dataset_file}")
    
    try:
        # Import training modules
        from dataset.harmonic_field_dataset import HarmonicFieldDataset
        from dataset.compute_stats import compute_global_stats
        from torch.utils.data import DataLoader
        
        # Load dataset
        data = torch.load(dataset_file, map_location='cpu', weights_only=False)
        print(f"✓ Loaded {len(data)} samples")
        
        # Compute global statistics
        mu, sigma = compute_global_stats(data[:min(100, len(data))])  # Use subset for speed
        print(f"✓ Global stats: μ={mu:.6f}, σ={sigma:.6f}")
        
        # Extract geometry and boundary masks from the first sample if available
        sample = data[0]
        if 'gmask' in sample and 'bmask' in sample:
            gmask = torch.tensor(sample['gmask'], dtype=torch.float32)
            bmask = torch.tensor(sample['bmask'], dtype=torch.float32)
            print(f"✓ Using masks from dataset samples")
        else:
            raise ValueError("Dataset samples lack 'gmask' or 'bmask' keys required for conditioning")
        
        # Create dataset
        dataset = HarmonicFieldDataset(data[:10], mu, sigma)  # Use small subset
        print(f"✓ Created HarmonicFieldDataset with {len(dataset)} samples")
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        u_batch, cond_batch = next(iter(dataloader))
        
        print(f"✓ DataLoader test:")
        print(f"  u_batch shape: {u_batch.shape}")
        print(f"  cond_batch shape: {cond_batch.shape}")
        print(f"  Expected: u=(B,1,H,W), cond=(B,4,H,W)")
        
        # Validate shapes
        B, C_u, H, W = u_batch.shape
        B2, C_cond, H2, W2 = cond_batch.shape
        
        assert B == B2 and H == H2 and W == W2, "Batch size or spatial dims mismatch"
        assert C_u == 1, f"Expected 1 channel for u, got {C_u}"
        assert C_cond == 4, f"Expected 4 channels for cond, got {C_cond}"
        
        print("✅ Training compatibility validated successfully!")
        
    except ImportError as e:
        print(f"❌ Missing training modules: {e}")
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")


def generate_sample_visualization(dataset_file, output_dir="sample_plots"):
    """Generate visualization of dataset samples"""
    print(f"\nGenerating sample visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Load dataset
        data = torch.load(dataset_file, map_location='cpu', weights_only=False)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize first few samples (5 samples starting from index 15)
        min_num_samples = 5
        start_idx       = 15
        num_samples     = min(min_num_samples, len(data))
        
        for i in range(start_idx, start_idx + num_samples):
            sample = data[i]
            
            _, axes = plt.subplots(1, 5, figsize=(25, 5))
            
            # Plot u field
            im1 = axes[0].imshow(sample['u'], cmap='RdBu_r', origin='lower')
            axes[0].set_title(f'Sample {i}: u(x,y)')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')
            plt.colorbar(im1, ax=axes[0])
            
            # Plot geometry mask
            im2 = axes[1].imshow(sample['gmask'], cmap='gray', origin='lower')
            axes[1].set_title(f'Sample {i}: Geometry Mask')
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('y')
            plt.colorbar(im2, ax=axes[1])
            
            # Plot boundary mask
            im3 = axes[2].imshow(sample['bmask'], cmap='binary', origin='lower')
            axes[2].set_title(f'Sample {i}: Boundary Mask')
            axes[2].set_xlabel('x')
            axes[2].set_ylabel('y')
            plt.colorbar(im3, ax=axes[2])
            
            # Plot Dirichlet BC
            im4 = axes[3].imshow(sample['dirichlet'], cmap='viridis', origin='lower')
            axes[3].set_title(f'Sample {i}: Dirichlet BC')
            axes[3].set_xlabel('x')
            axes[3].set_ylabel('y')
            plt.colorbar(im4, ax=axes[3])
            
            # Plot Neumann BC
            im5 = axes[4].imshow(sample['neumann'], cmap='plasma', origin='lower')
            axes[4].set_title(f'Sample {i}: Neumann BC')
            axes[4].set_xlabel('x')
            axes[4].set_ylabel('y')
            plt.colorbar(im5, ax=axes[4])
            
            plt.tight_layout()
            
            output_file = os.path.join(output_dir, f'sample_{i:03d}.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {output_file}")
        
        print(f"✓ Generated {num_samples} sample visualizations in {output_dir}/")
        
    except ImportError:
        print("❌ matplotlib not available for visualization")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Dataset utility tool")
    parser.add_argument("dataset_file", help="Path to dataset .pt file")
    parser.add_argument("--inspect", action="store_true", 
                       help="Inspect dataset structure and statistics")
    parser.add_argument("--validate", action="store_true",
                       help="Validate training pipeline compatibility")  
    parser.add_argument("--visualize", action="store_true",
                       help="Generate sample visualizations")
    parser.add_argument("--output_dir", default="sample_plots",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    if not any([args.inspect, args.validate, args.visualize]):
        # Default: run all checks
        args.inspect = True
        args.validate = True
        args.visualize = True
    
    print("=" * 60)
    print("HARMONIC DATASET UTILITY")
    print("=" * 60)
    
    if args.inspect:
        inspect_dataset(args.dataset_file)
    
    if args.validate:
        validate_training_compatibility(args.dataset_file)
    
    if args.visualize:
        generate_sample_visualization(args.dataset_file, args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ Dataset utility completed")
    print("=" * 60)


if __name__ == "__main__":
    main()