#!/usr/bin/env python3
"""
Grid Search for Diffusion Model Hyperparameters.

This script performs a comprehensive grid search to find the best hyperparameters
for the diffusion training stage (without physics loss). It searches over:
- batch_size: [32, 64, 128]
- learning_rate: [1e-3, 1e-4, 1e-5]
- pixel_res: [64, 128]
- num_epochs: runs 100 epochs and selects best checkpoint

The best combination is selected based on validation diffusion loss.
Since batch_size affects both dataset generation and training, the dataset
is regenerated for each batch_size value.

Usage:
    python model/grid_search_hyperparameters.py --num_samples 10000 --output_dir ./grid_search_results
"""

import os
import sys
import json
import csv
import subprocess
import argparse
import logging
from datetime import datetime
from pathlib import Path
import shutil

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GridSearchManager:
    """Manages grid search execution and result tracking."""
    
    def __init__(self, output_dir, num_samples=10000, dataset_seed=42,
                 batch_sizes=None, learning_rates=None, pixel_res=None,
                 num_epochs=100, save_every=5, device='auto', early_stopping_patience=10):
        """
        Initialize grid search manager.
        
        Args:
            output_dir: directory to store all results
            num_samples: number of samples to generate in dataset
            dataset_seed: random seed for dataset generation
            batch_sizes: list of batch sizes to try
            learning_rates: list of learning rates to try
            pixel_res: list of UNet2D sample sizes to try
            num_epochs: number of epochs to train each configuration
            save_every: save checkpoint every N epochs
            device: device to use for training ('auto', 'cuda', 'cpu')
            early_stopping_patience: number of epochs without improvement before stopping (default: 10)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_samples = num_samples
        self.dataset_seed = dataset_seed
        self.num_epochs = num_epochs
        self.save_every = save_every
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        
        # Grid search space
        self.batch_sizes = batch_sizes if batch_sizes is not None else [32, 64, 128]
        self.learning_rates = learning_rates if learning_rates is not None else [1e-3, 1e-4, 1e-5]
        self.pixel_res = pixel_res if pixel_res is not None else [64, 128]
        
        # Fixed parameters for diffusion-only training (no physics)
        self.fixed_params = {
            'grid_size': 3.0,
            'num_diffusion_timesteps': 1000,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'physics_weight_start': 0.0,
            'physics_weight_step': 0.0,
            'boundary_weight': 0.0,
            'neumann_weight': 0.0,
        }
        
        # Results tracking
        self.results = []  # List of dicts: {config_id, batch_size, lr, pixel_res, epoch_losses}
        self.best_config = None
        self.best_val_loss = 1e6
        
        logger.info(f"Grid Search Manager initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Dataset samples: {self.num_samples}")
        logger.info(f"Training epochs: {self.num_epochs}")
        logger.info(f"Batch sizes: {self.batch_sizes}")
        logger.info(f"Learning rates: {self.learning_rates}")
        logger.info(f"Sample sizes: {self.pixel_res}")
    
    def _find_best_epoch_with_early_stopping(self, epoch_losses):
        """
        Find the best epoch using early stopping logic.
        
        Args:
            epoch_losses: list of dicts with epoch data including 'val_diffusion_loss_mse'
        
        Returns:
            tuple: (best_epoch, best_val_loss, stopped_early, actual_epochs_trained)
            or None if no valid epochs found
        """
        # Filter epochs with validation loss
        epochs_with_val = [e for e in epoch_losses if 'val_diffusion_loss_mse' in e]
        if not epochs_with_val:
            logger.warning(f"No validation losses found in epoch data")
            return None
        
        # Initialize tracking variables
        best_epoch = epochs_with_val[0]['epoch']
        best_val_loss = epochs_with_val[0]['val_diffusion_loss_mse']
        epochs_without_improvement = 0
        stopped_early = False
        actual_epochs_trained = len(epochs_with_val)
        
        # Iterate through epochs to find best and check early stopping
        for epoch_data in epochs_with_val:
            current_val_loss = epoch_data['val_diffusion_loss_mse']
            
            if current_val_loss < best_val_loss:
                # New best found, reset counter
                best_val_loss = current_val_loss
                best_epoch = epoch_data['epoch']
                epochs_without_improvement = 0
            else:
                # No improvement
                epochs_without_improvement += 1
                
                # Check if we should stop early
                if epochs_without_improvement >= self.early_stopping_patience:
                    stopped_early = True
                    break
        
        # Log result
        if stopped_early:
            logger.info(f"✓ Best epoch: {best_epoch} with val_diffusion_loss={best_val_loss:.6f} (early stopped at epoch {epoch_data['epoch']})")
        else:
            logger.info(f"✓ Best epoch: {best_epoch} with val_diffusion_loss={best_val_loss:.6f} (trained all {actual_epochs_trained} epochs)")
        
        return best_epoch, best_val_loss, stopped_early, actual_epochs_trained

    
    def generate_dataset(self, pixel_res):
        """
        Generate dataset for specific pixel_res.
        
        Args:
            pixel_res: grid resolution (pixel_res)
        
        Returns:
            paths to train and val dataset files
        """
        logger.info(f"Generating dataset for pixel_res={pixel_res}")
        
        # Create dataset directory for this configuration (only depends on pixel_res)
        dataset_dir = self.output_dir / f"datasets/res{pixel_res}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_base = dataset_dir / f"harmonic_dataset_res{pixel_res}"
        
        # Check if dataset already exists
        train_file = f"{dataset_base}_train.pt"
        val_file = f"{dataset_base}_val.pt"
        metadata_file = f"{dataset_base}_metadata.pt"
        
        if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(metadata_file):
            logger.info(f"Dataset already exists, skipping generation: {dataset_base}")
            return train_file, val_file, metadata_file
        
        # Generate dataset using the existing script
        cmd = [
            sys.executable, 'dataset/generate_harmonic_dataset.py',
            '--num_samples', str(self.num_samples),
            '--pixel_res', str(pixel_res),
            '--N', '8',
            '--output', f"{dataset_base}.pt",
            '--seed', str(self.dataset_seed),
            '--val_split', '0.2'
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Dataset generation completed successfully")
            logger.debug(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Dataset generation failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise
        
        return train_file, val_file, metadata_file
    
    def train_configuration(self, batch_size, learning_rate, pixel_res, 
                          train_dataset, val_dataset, config_id):
        """
        Train a specific configuration.
        
        Args:
            batch_size: training batch size
            learning_rate: learning rate
            pixel_res: grid resolution
            train_dataset: path to training dataset
            val_dataset: path to validation dataset
            config_id: unique identifier for this configuration
        
        Returns:
            dictionary with training results
        """
        logger.info(f"Training configuration: {config_id}")
        logger.info(f"  batch_size={batch_size}, lr={learning_rate}, pixel_res={pixel_res}")
        
        # Create checkpoint directory for this configuration
        checkpoint_dir = self.output_dir / f"checkpoints/{config_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Build training command
        cmd = [
            'python', 'model/physics_training_loop.py',
            '--dataset', train_dataset,
            '--val_dataset', val_dataset,
            '--num_epochs', str(self.num_epochs),
            '--batch_size', str(batch_size),
            '--learning_rate', str(learning_rate),
            '--pixel_res', str(pixel_res),
            '--grid_size', str(self.fixed_params['grid_size']),
            '--num_diffusion_timesteps', str(self.fixed_params['num_diffusion_timesteps']),
            '--beta_start', str(self.fixed_params['beta_start']),
            '--beta_end', str(self.fixed_params['beta_end']),
            '--physics_weight_start', str(self.fixed_params['physics_weight_start']),
            '--physics_weight_step', str(self.fixed_params['physics_weight_step']),
            '--boundary_weight', str(self.fixed_params['boundary_weight']),
            '--neumann_weight', str(self.fixed_params['neumann_weight']),
            '--output_dir', str(checkpoint_dir),
            '--save_every', str(self.save_every),
            '--device', self.device
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Run training and capture logs
        log_file = checkpoint_dir / 'training.log'
        
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            logger.info(f"Training completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            with open(log_file, 'r') as f:
                logger.error(f"Training log:\n{f.read()}")
            raise
        
        # Load epoch losses from JSON output (generated by physics_training_loop.py as training_epoch_losses.json)
        output_json = checkpoint_dir / 'training_epoch_losses.json'
        if not output_json.exists():
            logger.warning(f"Epoch losses JSON not found: {output_json}")
            return None
        
        with open(output_json, 'r') as f:
            epoch_data = json.load(f)
        
        epoch_losses = epoch_data.get('epoch_losses', [])
        
        if not epoch_losses:
            logger.warning(f"No epoch losses found in JSON output")
            return None
        
        # Find best validation epoch with early stopping
        result_tuple = self._find_best_epoch_with_early_stopping(epoch_losses)
        if result_tuple is None:
            return None
        
        best_epoch, best_val_loss, stopped_early, actual_epochs_trained = result_tuple
        
        # Prepare result with all epoch data
        result = {
            'config_id': config_id,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'pixel_res': pixel_res,
            'epoch_losses': epoch_losses,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'stopped_early': stopped_early,
            'actual_epochs_trained': actual_epochs_trained,
            'checkpoint_path': str(checkpoint_dir / f"cond_ddpm_epoch{best_epoch}"),
            'dataset_path': train_dataset,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
 
    def run_grid_search(self):
        """
        Execute complete grid search.
        
        Nested loop structure:
        1. For each pixel_res: generate dataset once
        2. For each batch_size: use same dataset
        3. For each learning_rate: train model
        """
        logger.info("=" * 80)
        logger.info("STARTING GRID SEARCH")
        logger.info("=" * 80)
        
        total_configs = len(self.batch_sizes) * len(self.pixel_res) * len(self.learning_rates)
        config_counter = 0
        
        # Outer loop: pixel_res (requires dataset regeneration)
        for pixel_res in self.pixel_res:
            logger.info(f"\n{'='*80}")
            logger.info(f"PIXEL RESOLUTION: {pixel_res}")
            logger.info(f"{'='*80}")
            
            # Generate dataset once per pixel_res
            try:
                train_dataset, val_dataset, metadata = self.generate_dataset(pixel_res)
            except Exception as e:
                logger.error(f"Failed to generate dataset for pixel_res={pixel_res}: {e}")
                continue
            
            # Middle loop: batch_size (uses same dataset)
            for batch_size in self.batch_sizes:
                logger.info(f"\n{'-'*80}")
                logger.info(f"BATCH SIZE: {batch_size}")
                logger.info(f"{'-'*80}")
                
                # Inner loop: learning_rate
                for learning_rate in self.learning_rates:
                    config_counter += 1
                    config_id = f"bs{batch_size}_lr{learning_rate}_res{pixel_res}"
                    
                    logger.info(f"\n{'*'*60}")
                    logger.info(f"Configuration {config_counter}/{total_configs}: {config_id}")
                    logger.info(f"{'*'*60}")
                    
                    try:
                        # Train configuration
                        result = self.train_configuration(
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            pixel_res=pixel_res,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            config_id=config_id
                        )
                        
                        if result is None:
                            logger.warning(f"No results for configuration {config_id}")
                            continue
                        
                        # Store result
                        self.results.append(result)
                        
                        # Update best configuration
                        if result['best_val_loss'] < self.best_val_loss:
                            self.best_val_loss = result['best_val_loss']
                            self.best_config = result
                            logger.info(f"🎉 NEW BEST CONFIGURATION: {config_id}")
                            logger.info(f"   Val Diffusion Loss: {self.best_val_loss:.6f}")
                        
                        # Log progress
                        logger.info(f"✓ Configuration {config_counter}/{total_configs} completed")
                        logger.info(f"  Val Diffusion Loss: {result['best_val_loss']:.6f}")
                        logger.info(f"  Best Epoch: {result['best_epoch']}")
                        
                    except Exception as e:
                        logger.error(f"Failed to train configuration {config_id}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        continue
        
        logger.info(f"\n{'='*80}")
        logger.info("GRID SEARCH COMPLETED")
        logger.info(f"{'='*80}")
        
        # Save all epoch losses to CSV
        self._save_epoch_losses_csv()
        
        # Print summary to console
        self._print_summary()
        
        # Generate visualizations
        self._generate_visualizations()
        
        return self.best_config
    
    def _save_epoch_losses_csv(self):
        """Save all epoch losses from all configurations to a single CSV file."""
        csv_file = self.output_dir / 'all_epoch_losses.csv'
        
        logger.info(f"Writing epoch losses to: {csv_file}")
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'config_id', 'batch_size', 'learning_rate', 'pixel_res',
                'epoch', 'train_diffusion_loss_mse', 'train_physics_loss_mse', 'train_total_loss_mse',
                'val_diffusion_loss_mse', 'val_physics_loss_mse', 'val_total_loss_mse', 'physics_weight'
            ])
            writer.writeheader()
            
            for result in self.results:
                config_id = result['config_id']
                batch_size = result['batch_size']
                learning_rate = result['learning_rate']
                pixel_res = result['pixel_res']
                
                for epoch_data in result['epoch_losses']:
                    row = {
                        'config_id': config_id,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'pixel_res': pixel_res,
                        'epoch': epoch_data['epoch'],
                        'train_diffusion_loss_mse': epoch_data['train_diffusion_loss_mse'],
                        'train_physics_loss_mse': epoch_data['train_physics_loss_mse'],
                        'train_total_loss_mse': epoch_data['train_total_loss_mse'],
                        'val_diffusion_loss_mse': epoch_data.get('val_diffusion_loss_mse', ''),
                        'val_physics_loss_mse': epoch_data.get('val_physics_loss_mse', ''),
                        'val_total_loss_mse': epoch_data.get('val_total_loss_mse', ''),
                        'physics_weight': epoch_data['physics_weight']
                    }
                    writer.writerow(row)
        
        logger.info(f"Saved {sum(len(r['epoch_losses']) for r in self.results)} epoch records to CSV")
    
    def _print_summary(self):
        """Print grid search summary to console."""
        logger.info(f"\n{'='*80}")
        logger.info("GRID SEARCH SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total configurations tested: {len(self.results)}")
        
        if self.best_config is None:
            logger.warning("No successful configurations found")
        else:
            logger.info(f"\nBest Configuration:")
            logger.info(f"  Config ID: {self.best_config['config_id']}")
            logger.info(f"  Batch Size: {self.best_config['batch_size']}")
            logger.info(f"  Learning Rate: {self.best_config['learning_rate']}")
            logger.info(f"  Sample Size: {self.best_config['pixel_res']}")
            logger.info(f"  Best Epoch: {self.best_config['best_epoch']}")
            logger.info(f"  Val Diffusion Loss: {self.best_config['best_val_loss']:.6f}")
            logger.info(f"  Early Stopped: {'Yes' if self.best_config.get('stopped_early', False) else 'No'}")
            logger.info(f"  Epochs Trained: {self.best_config.get('actual_epochs_trained', 'N/A')}")
            logger.info(f"  Checkpoint: {self.best_config['checkpoint_path']}")
        
        logger.info(f"{'='*80}")
    
    def _generate_visualizations(self):
        """Generate visualization plots for grid search results."""
        if len(self.results) == 0:
            logger.warning("No results to visualize")
            return
        
        logger.info("Generating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Grid Search Results: Diffusion Model Hyperparameters', fontsize=16, fontweight='bold')
        
        # Extract data (compute on-the-fly from epoch_losses)
        batch_sizes = [r['batch_size'] for r in self.results]
        learning_rates = [r['learning_rate'] for r in self.results]
        pixel_res = [r['pixel_res'] for r in self.results]
        val_losses = [r['best_val_loss'] for r in self.results]
        best_epochs = [r['best_epoch'] for r in self.results]
        
        # 1. Val Loss vs Batch Size
        ax = axes[0, 0]
        for lr in self.learning_rates:
            for pixel_res in self.pixel_res:
                mask = [(r['learning_rate'] == lr and r['pixel_res'] == pixel_res) for r in self.results]
                bs = [self.results[i]['batch_size'] for i, m in enumerate(mask) if m]
                vl = [self.results[i]['best_val_loss'] for i, m in enumerate(mask) if m]
                if bs:
                    ax.plot(bs, vl, marker='o', label=f'LR={lr}, PX={pixel_res}')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Validation Diffusion Loss')
        ax.set_title('Loss vs Batch Size')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. Val Loss vs Learning Rate (log scale)
        ax = axes[0, 1]
        for bs in self.batch_sizes:
            for pixel_res in self.pixel_res:
                mask = [(r['batch_size'] == bs and r['pixel_res'] == pixel_res) for r in self.results]
                lr = [self.results[i]['learning_rate'] for i, m in enumerate(mask) if m]
                vl = [self.results[i]['best_val_loss'] for i, m in enumerate(mask) if m]
                if lr:
                    ax.plot(lr, vl, marker='o', label=f'BS={bs}, PX={pixel_res}')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Validation Diffusion Loss')
        ax.set_title('Loss vs Learning Rate')
        ax.set_xscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3. Val Loss vs Sample Size
        ax = axes[0, 2]
        for bs in self.batch_sizes:
            for lr in self.learning_rates:
                mask = [(r['batch_size'] == bs and r['learning_rate'] == lr) for r in self.results]
                pixel_res = [self.results[i]['pixel_res'] for i, m in enumerate(mask) if m]
                vl = [self.results[i]['best_val_loss'] for i, m in enumerate(mask) if m]
                if pixel_res:
                    ax.plot(pixel_res, vl, marker='o', label=f'BS={bs}, LR={lr}')
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Validation Diffusion Loss')
        ax.set_title('Loss vs Sample Size')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 4. Best Epoch vs Configuration
        ax = axes[1, 0]
        config_labels = [f"{r['batch_size']}\n{r['learning_rate']:.0e}\n{r['pixel_res']}" 
                        for r in self.results]
        x_pos = np.arange(len(config_labels))
        colors = ['green' if r == self.best_config else 'blue' for r in self.results]
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
                losses = [r['best_val_loss'] for r in self.results 
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
        
        # Add text annotations
        for i in range(len(unique_lr)):
            for j in range(len(unique_bs)):
                if not np.isnan(heatmap_data[i, j]):
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                                 ha="center", va="center", color="white", fontsize=8)
        
        # 6. Overall ranking
        ax = axes[1, 2]
        sorted_results = sorted(self.results, key=lambda x: x['best_val_loss'])
        config_labels_sorted = [f"{r['config_id']}" for r in sorted_results[:10]]
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
        
        # Save figure
        plot_file = self.output_dir / 'grid_search_results.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to: {plot_file}")
        
        plt.close()


def main():
    """Main entry point for grid search script."""
    
    parser = argparse.ArgumentParser(
        description='Grid search for diffusion model hyperparameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python model/grid_search_hyperparameters.py --num_samples 10000 --output_dir ./grid_search_results
  
This will search over:
  - batch_size: [32, 64, 128]
  - learning_rate: [1e-3, 1e-4, 1e-5]
  - pixel_res: [64, 128]
  - num_epochs: 100 (with checkpoint selection)

The best configuration is selected based on validation diffusion loss.
        """
    )
    
    # Dataset parameters
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples in dataset (default: 10000)')
    parser.add_argument('--dataset_seed', type=int, default=42,
                       help='Random seed for dataset generation (default: 42)')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs per configuration (default: 100)')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device for training (default: auto)')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Number of epochs without improvement before early stopping (default: 10)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./grid_search_results',
                       help='Output directory for results (default: ./grid_search_results)')
    
    # Grid search space (optional overrides)
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=None,
                       help='Batch sizes to search (default: [32, 64, 128])')
    parser.add_argument('--learning_rates', type=float, nargs='+', default=None,
                       help='Learning rates to search (default: [1e-3, 1e-4, 1e-5])')
    parser.add_argument('--pixel_res', type=int, nargs='+', default=None,
                       help='Pixel resolutions to search (default: [64, 128])')
    
    args = parser.parse_args()
    
    # Initialize grid search manager
    manager = GridSearchManager(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        dataset_seed=args.dataset_seed,
        num_epochs=args.num_epochs,
        batch_sizes=args.batch_sizes,
        learning_rates=args.learning_rates,
        pixel_res=args.pixel_res,
        save_every=args.save_every,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
    )
    
    # Run grid search
    try:
        best_config = manager.run_grid_search()
        
        if best_config:
            logger.info(f"\n{'='*80}")
            logger.info("🎉 GRID SEARCH SUCCESSFUL!")
            logger.info(f"{'='*80}")
            logger.info(f"\nBest Configuration Found:")
            logger.info(f"  Config: {best_config['config_id']}")
            logger.info(f"  Validation Diffusion Loss: {best_config['best_val_loss']:.6f}")
            logger.info(f"  Best Checkpoint: {best_config['checkpoint_path']}")
            logger.info(f"\nTo use this checkpoint for physics training, run:")
            logger.info(f"  python model/physics_training_loop.py \\")
            logger.info(f"    --dataset {best_config['dataset_path']} \\")
            logger.info(f"    --batch_size {best_config['batch_size']} \\")
            logger.info(f"    --learning_rate {best_config['learning_rate']} \\")
            logger.info(f"    --pixel_res {best_config['pixel_res']} \\")
            logger.info(f"    --resume_from_checkpoint {best_config['checkpoint_path']} \\")
            logger.info(f"    --physics_weight_start 0.1 \\")
            logger.info(f"    --physics_weight_step 0.01")
            logger.info(f"{'='*80}\n")
        else:
            logger.error("Grid search failed to find best configuration")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\nGrid search interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Grid search failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
