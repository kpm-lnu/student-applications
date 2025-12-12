# Model Training and Sample Generation

This directory contains scripts for training physics-informed diffusion models to solve inverse problems for harmonic functions using Denoising Diffusion Probabilistic Models (DDPM).

## Overview

The model learns to generate harmonic functions that satisfy:

- **Laplace equation**: ∇²u = 0 in the interior domain
- **Boundary conditions**:
  - Dirichlet conditions (u = g on boundary)
  - Neumann conditions (∇u·n = h on boundary)

The training uses a conditional DDPM with physics-informed losses to ensure generated solutions are physically meaningful.

## Files

### 1. `physics_training_loop.py`

Main training script for physics-informed diffusion models. Implements:

- Conditional UNet2D architecture (5 input channels: 1 noisy field + 4 conditioning maps)
- DDPM forward/reverse diffusion process
- Physics-informed loss combining:
  - Diffusion loss (noise prediction)
  - Laplace equation loss (∇²u = 0)
  - Boundary condition losses (Dirichlet & Neumann)
- Automatic validation: Runs validation phase after each training epoch
- Auto-detection: Automatically finds validation dataset file
- **Early Stopping**: Optional patience-based early stopping on validation loss with configurable metric (physics L2 error or diffusion MSE loss, disabled by default)
- **Structured output**: Outputs epoch-level training/validation losses to JSON file
- **In-memory tracking**: Returns all epoch losses for programmatic access

### 2. `grid_search_hyperparameters.py`

Automated hyperparameter optimization script that performs comprehensive grid search to find the best training configuration. **Streamlined and efficient** design:

- **Search Space**:
  - `batch_size`: [32, 64, 128]
  - `learning_rate`: [1e-3, 1e-4, 1e-5]
  - `pixel_res`: [64, 128]
  - `num_epochs`: 100 (automatically selects best checkpoint)
- **Smart Dataset Management**: Regenerates datasets for each batch_size × pixel_res combination
- **Pure Diffusion Focus**: Trains without physics loss to optimize base diffusion performance
- **Early Stopping**: Monitors validation loss and stops training if no improvement for N consecutive epochs (configurable patience, default: 10)
- **In-Memory Processing**: Keeps all epoch data in memory during execution (no log file parsing)
- **Structured Data Capture**: Receives epoch losses from training via JSON output
- **Best Model Selection**: Chooses configuration based on minimum validation diffusion loss
- **Single Comprehensive CSV**: Outputs all epoch losses from all configurations to `all_epoch_losses.csv`
- **Full Training Logs**: Preserves complete training logs in `training.log` files per configuration
- **Rich Visualizations**: 6-panel plots showing loss curves and configuration rankings
- **Robust Execution**: Continues even if individual configurations fail

### 3. `generate_samples.py`

Inference module for generating samples from trained models:

- Loads trained model and scheduler from checkpoints
- Performs iterative denoising from pure noise
- Enforces boundary conditions on final outputs

## Usage

#### Finding the Best Model

The grid search automatically identifies the best configuration by:

1. For each configuration:
   - Monitors validation loss after each epoch
   - Stops early if no improvement for `early_stopping_patience` consecutive epochs
   - Selects the epoch with minimum validation diffusion loss
2. Comparing best validation losses across all configurations
3. Selecting the overall winner

The best configuration is reported in the console output (including whether it stopped early and how many epochs it actually trained) and highlighted in visualizations. You can also manually analyze the `all_epoch_losses.csv` file to:

- Plot training curves for specific configurations
- Identify overfitting by comparing train vs validation losses
- Choose alternative configurations based on different criteria
- See exactly which epoch each configuration stopped at

#### Using Grid Search Results

Before training a full model, use grid search to find the best hyperparameters based on validation diffusion loss.

#### Basic Grid Search

Run with default settings (searches over batch_size, learning_rate, and pixel_res):

```bash
python model/grid_search_hyperparameters.py \
    --num_samples 10000 \
    --output_dir ./grid_search_results
```

This will:

1. Test all combinations of batch_size [32, 64, 128], learning_rate [1e-3, 1e-4, 1e-5], and pixel_res [64, 128]
2. Generate datasets as needed for each batch_size × pixel_res combination
3. Train each configuration for 100 epochs with diffusion loss only (no physics)
4. Select the best checkpoint for each configuration based on validation loss
5. Identify the overall best configuration
6. Generate comprehensive results with CSV logs and visualizations

#### Custom Grid Search

Customize the search space and parameters:

```bash
python model/grid_search_hyperparameters.py \
    --num_samples 50000 \
    --num_epochs 100 \
    --early_stopping_patience 15 \
    --batch_sizes 32 64 \
    --learning_rates 1e-3 1e-4 1e-5 \
    --pixel_res_list 64 128 \
    --output_dir ./grid_search_results \
    --save_every 5 \
    --device cuda
```

#### Grid Search Parameters

**Dataset Parameters:**

- `--num_samples`: Number of samples in dataset (default: `10000`)
- `--dataset_seed`: Random seed for dataset generation (default: `42`)

**Training Parameters:**

- `--num_epochs`: Number of training epochs per configuration (default: `100`)
- `--save_every`: Save checkpoint every N epochs (default: `5`)
- `--device`: Device for training - `auto`, `cuda`, or `cpu` (default: `auto`)
- `--early_stopping_patience`: Number of epochs without improvement before early stopping (default: `10`)

**Grid Search Space (optional overrides):**

- `--batch_sizes`: List of batch sizes to search (default: `[32, 64, 128]`)
- `--learning_rates`: List of learning rates to search (default: `[1e-3, 1e-4, 1e-5]`)
- `--pixel_res_list`: List of pixel resolutions to search (default: `[64, 128]`)

**Output:**

- `--output_dir`: Output directory for all results (default: `./grid_search_results`)

#### Grid Search Output

The script generates a streamlined output structure:

```
grid_search_results/
├── all_epoch_losses.csv                # ALL epoch losses from ALL configurations (comprehensive)
├── grid_search_results.png             # 6-panel visualization
├── datasets/                           # Generated datasets
│   ├── bs32_ss64/
│   │   ├── harmonic_dataset_bs32_ss64_train.pt
│   │   ├── harmonic_dataset_bs32_ss64_val.pt
│   │   └── harmonic_dataset_bs32_ss64_metadata.pt
│   └── ...
└── checkpoints/                        # All trained models
    ├── bs32_lr0.001_ss64/
    │   ├── training.log                # Full training output
    │   ├── training_epoch_losses.json  # Structured epoch data
    │   ├── cond_ddpm_epoch5/
    │   ├── cond_ddpm_epoch10/
    │   └── ...
    └── ...
```

The **`all_epoch_losses.csv`** file contains one row per epoch per configuration with columns:

- **Configuration**: `config_id`, `batch_size`, `learning_rate`, `pixel_res`
- **Epoch info**: `epoch`
- **Training metrics**: `train_diffusion_loss`, `train_physics_loss`, `train_total_loss`
- **Validation metrics**: `val_diffusion_loss`, `val_physics_loss`, `val_total_loss`
- **Other**: `physics_weight`

This single CSV provides a complete training history for all configurations, making it easy to:

- Analyze training curves across all hyperparameter combinations
- Compare convergence rates between configurations
- Identify the optimal epoch for each configuration (where early stopping occurred or minimum validation loss)
- Detect overfitting patterns manually if needed

**Note on Early Stopping**: Configurations may train for fewer than `num_epochs` if validation loss doesn't improve for `early_stopping_patience` consecutive epochs. The CSV will only contain rows for epochs that were actually executed.

**Visualizations include:**

1. Validation loss vs batch size
2. Validation loss vs learning rate (log scale)
3. Validation loss vs sample size
4. Best epoch per configuration
5. Heatmap of learning rate vs batch size (averaged over sample sizes)
6. Top 10 configurations ranking

The visualization automatically highlights the best configuration in green.

#### Finding the Best Model

After grid search completes, use the best configuration for physics training:

```bash
# The script will print the exact command, for example:
python model/physics_training_loop.py \
    --dataset grid_search_results/datasets/bs64_ss128/harmonic_dataset_bs64_ss128_train.pt \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --pixel_res 128 \
    --resume_from_checkpoint grid_search_results/checkpoints/bs64_lr0.0001_ss128/cond_ddpm_epoch50 \
    --physics_weight_start 0.1 \
    --physics_weight_step 0.01
```

### Training a Model

#### Basic Training

Train for 20 epochs with default parameters (automatically includes validation):

```bash
python model/physics_training_loop.py \
    --dataset harmonic_field_dataset_train.pt \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --pixel_res 64 \
    --grid_size 3.0 \
    --num_diffusion_timesteps 1000 \
    --beta_start 0.0001 \
    --beta_end 0.02 \
    --physics_weight_start 0.1 \
    --physics_weight_step 0.01 \
    --boundary_weight 2.0 \
    --neumann_weight 0.5 \
    --output_dir ./checkpoints \
    --save_every 5 \
    --resume_from_checkpoint None \
    --device auto \
    --early_stopping_patience 10
```

**Note**: The validation dataset is auto-detected (looks for `*_val.pt` matching the training file). You can explicitly specify it with `--val_dataset` if needed.

#### Parameters Explained

**Dataset & I/O:**

- `--dataset`: Path to training dataset file (default: `harmonic_field_dataset_train.pt`)
- `--val_dataset`: Path to validation dataset file (default: auto-detected from training file)
- `--output_dir`: Directory to save model checkpoints (default: `./checkpoints`)
- `--save_every`: Save checkpoint every N epochs (default: `5`)
- `--resume_from_checkpoint`: Path to checkpoint directory to resume from (default: `None`)
- `--device`: Device for training - `auto`, `cuda`, or `cpu` (default: `auto`)

**Training Hyperparameters:**

- `--num_epochs`: Number of training epochs (default: `5`)
- `--batch_size`: Training batch size (default: `16`)
- `--learning_rate`: Adam optimizer learning rate (default: `1e-4`)
- `--early_stopping_patience`: Number of epochs without improvement before stopping (default: `None`, disabled)
- `--early_stopping_metric`: Metric to use for early stopping - `physics` (L2 error) or `diffusion` (MSE) (default: `physics`)

**Model Configuration:**

- `--pixel_res`: Grid resolution H×W (default: `64`)
- `--grid_size`: Physical domain size, e.g., 3.0 for [-1.5,1.5]×[-1.5,1.5] (default: `3.0`)
- `--num_diffusion_timesteps`: Number of diffusion timesteps T (default: `1000`)
- `--beta_start`: Starting noise variance β₁ (default: `0.0001`)
- `--beta_end`: Ending noise variance βₜ (default: `0.02`)

**Physics Loss Weights:**

- `--physics_weight_start`: Initial physics loss weight (default: `0.1`)
- `--physics_weight_step`: Increment per epoch for curriculum learning (default: `0.01`)
- `--boundary_weight`: Weight for boundary condition losses (default: `2.0`)
- `--neumann_weight`: Relative weight for Neumann vs Dirichlet losses (default: `0.5`)

**Note on Physics Loss Computation:** The physics losses (Laplace, Dirichlet, and Neumann) are computed using **MSE (Mean Squared Error)** for training, which provides smoother gradients and better convergence. Additionally, **L∞ (max absolute error) metrics** are logged for interpretability, showing the worst-case pointwise violation of physical constraints.

#### Resume Training from Checkpoint

To continue training from epoch 15:

```bash
python model/physics_training_loop.py \
    --dataset harmonic_field_dataset_train.pt \
    --num_epochs 10 \
    --resume_from_checkpoint ./checkpoints/cond_ddpm_epoch15 \
    --output_dir ./checkpoints
```

The script will automatically detect the epoch number and continue from there. Validation will continue to run after each epoch.

### Generating Samples

Use the `generate_samples.py` module in Python:

```python
import torch
from model.generate_samples import load_trained_model, generate_samples

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
checkpoint_path = "./checkpoints/cond_ddpm_epoch20"
scheduler_config_path = "./checkpoints/scheduler_epoch20.json/scheduler_config.json"

model, scheduler = load_trained_model(
    checkpoint_path=checkpoint_path,
    scheduler_config_path=scheduler_config_path,
    device=device
)

# Prepare conditioning (geometry, boundary, Dirichlet, Neumann)
# cond shape: [batch_size, 4, height, width]
cond = ...  # Your conditioning maps

# Load normalization stats from dataset metadata
metadata = torch.load('harmonic_field_dataset_metadata.pt')
mu = metadata['global_mu']
sigma = metadata['global_sigma']

# Generate samples
samples = generate_samples(
    model=model,
    scheduler=scheduler,
    cond=cond,
    mu=mu,
    sigma=sigma,
    denoising_steps=1000,
    device=device
)

# samples shape: [batch_size, 1, height, width]
```

## Model Architecture

### UNet2D Configuration

- **Input channels**: 5
  - 1 channel: noisy harmonic function u
  - 4 channels: conditioning (geometry mask, boundary mask, Dirichlet BC, Neumann BC)
- **Output channels**: 1 (predicted noise)
- **Layers per block**: 2
- **Block output channels**: (64, 128, 256)
- **Architecture**: 3 down blocks + 3 up blocks

### DDPM Process

**Forward (noising):**

```
q(xₜ|x₀) = 𝒩(xₜ; √ᾱₜ x₀, (1-ᾱₜ)I)
where ᾱₜ = ∏ᵢ₌₁ᵗ(1-βᵢ)
```

**Reverse (denoising):**

```
pθ(xₜ₋₁|xₜ) = 𝒩(xₜ₋₁; μθ(xₜ,t), Σθ(xₜ,t))
where μθ = 1/√αₜ (xₜ - (1-αₜ)/√(1-ᾱₜ) εθ(xₜ,t))
```

## Physics Loss Components

The total loss combines diffusion and physics objectives:

```
L_total = L_diffusion + λ_physics * L_physics
```

Where:

- **L_diffusion**: MSE between predicted and actual noise
- **L_physics**: Combination of:
  - **L_laplace**: MSE of ∇²u in interior (should be ≈ 0)
  - **L_dirichlet**: MSE of (u - g) at boundary points
  - **L_neumann**: MSE of (∇u·n - h) at boundary points
  - **L_boundary** = L_dirichlet + λ_neumann \* L_neumann
  - **L_physics** = L_laplace + λ_boundary \* L_boundary

The physics weight λ_physics increases during training (curriculum learning).

**Loss Computation Details:**

- **Training loss**: Uses **MSE (Mean Squared Error)** for all physics losses with **per-sample computation**
  - Per-sample MSE: Each sample's squared errors are divided by its own number of active points
  - Batch averaging: Final loss is the mean of all per-sample MSEs
  - Ensures each sample contributes equally regardless of its geometry
  - Provides smoother gradients (no singularity at zero)
  - Penalizes large errors more heavily (quadratic)
  - Scale-invariant across different batch/grid sizes
  - Consistent with diffusion loss
- **Monitoring metrics**: Both **L∞ (absolute)** and **L∞ (relative)** are logged with **per-sample computation**
  - Per-sample L∞: Max error computed independently for each sample in batch
  - Batch averaging: Reported metric is mean of per-sample L∞ values
  - Absolute L∞: Direct max error value (e.g., 0.05)
  - Relative L∞: Max error normalized by each sample's own magnitude (e.g., 0.025 = 2.5%)
  - Directly interpretable: "average max violation across batch is 2.5% of solution magnitude"
  - Shows worst-case pointwise behavior per sample
  - Essential for understanding per-sample physics satisfaction
  - Scale-invariant comparison across different samples/datasets
  - Consistent metrics regardless of batch composition

## Output Structure

After training, checkpoints are saved in the following structure:

```
checkpoints/
├── cond_ddpm_epoch5/
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
│   └── diffusion_pytorch_model.safetensors
├── cond_ddpm_epoch10/
│   └── ...
├── scheduler_epoch5.json/
│   └── scheduler_config.json
└── scheduler_epoch10.json/
    └── scheduler_config.json
```

## Recommended Workflow

For best results, follow this workflow:

1. **Grid Search** (Optional but Recommended): Find optimal hyperparameters

   ```bash
   python model/grid_search_hyperparameters.py \
      --num_samples 25000 \
      --num_epochs 100 \
      --early_stopping_patience 10 \
      --save_every 1000 \
      --device auto \
      --output_dir ./grid_search_results
   ```

   **Note**: Early stopping will terminate training for each configuration if validation loss doesn't improve for 10 consecutive epochs, saving time on configurations that have converged.

2. **Diffusion Pre-training**: Train with best hyperparameters (no physics)

   ```bash
   python model/physics_training_loop.py \
       --dataset [best_dataset] \
       --batch_size [best_batch_size] \
       --learning_rate [best_learning_rate] \
       --pixel_res [best_pixel_res] \
       --physics_weight_start 0.0 \
       --physics_weight_step 0.0 \
       --num_epochs 50
   ```

3. **Physics Fine-tuning**: Resume from best checkpoint and add physics
   ```bash
   python model/physics_training_loop.py \
       --dataset [best_dataset] \
       --resume_from_checkpoint [best_checkpoint] \
       --physics_weight_start 0.1 \
       --physics_weight_step 0.01 \
       --num_epochs 50 \
       --early_stopping_patience 15 \
       --early_stopping_metric physics
   ```

Or use the curriculum training script to automate stages 2-3:

```bash
python model/curriculum_training.py \
    --dataset [best_dataset] \
    --batch_size [best_batch_size] \
    --learning_rate [best_learning_rate] \
    --pixel_res [best_pixel_res]
```

## Training Tips

1. **Run grid search first**: Use `grid_search_hyperparameters.py` to find optimal hyperparameters before investing in long training runs
2. **Use early stopping in grid search**: The default patience of 10 epochs balances exploration with efficiency; increase for more thorough search, decrease for faster results
3. **Enable early stopping for long runs**: Use `--early_stopping_patience` (e.g., 10-20) with `--early_stopping_metric physics` when training for many epochs to prevent overfitting and save compute time
4. **Start with pure diffusion**: Set `physics_weight_start=0.0` for initial epochs
5. **Gradual physics introduction**: Use small `physics_weight_step` (0.01-0.05)
6. **Adjust boundary weight**: Higher values (2.0-5.0) enforce boundary conditions more strongly
7. **Monitor losses**: Check that Laplace loss decreases and boundary losses remain low; watch relative L∞ metrics (aim for < 5% for good physics satisfaction)
8. **Use curriculum training**: The 3-stage approach generally produces better results
9. **Checkpoint regularly**: Save every 5 epochs to avoid losing progress
10. **Watch validation metrics**: If validation loss diverges from training, consider reducing learning rate or adjusting physics weight schedule
11. **Use train/val split**: Always use separate validation set to monitor generalization (default: auto-detected)
12. **Leverage grid search results**: The CSV and visualizations help understand hyperparameter sensitivity; check which configurations stopped early vs trained fully
13. **Interpret early stopping**: If training stops early consistently, consider increasing patience or adjusting learning rates; if it never stops early, you may be over-training

## Requirements

The training scripts require:

- PyTorch ≥ 2.0
- diffusers (Hugging Face)
- tqdm (progress bars)
- Custom modules from this repository:
  - `dataset.harmonic_field_dataset`
  - `grid.compute_grid`
  - `harmonic.kernels`
  - `harmonic.sample_harmonic_field`

See `../requirements.txt` for complete dependencies.

## Troubleshooting

### OOM (Out of Memory) Errors

- Reduce `--batch_size` (try 8 or 4)
- Reduce `--pixel_res` (try 32 instead of 64)
- Use mixed precision training (requires code modification)

### High Physics Loss

- Check dataset normalization (should use global μ and σ)
- Increase `--physics_weight_step` more gradually
- Ensure boundary conditions are properly specified in dataset

### Poor Boundary Condition Satisfaction

- Increase `--boundary_weight` (try 5.0 or 10.0)
- Adjust `--neumann_weight` if Neumann conditions are problematic
- Verify boundary masks are correctly defined in conditioning

### Training Instability

- Reduce `--learning_rate` (try 5e-5)
- Enable gradient clipping (already implemented at max_norm=1.0)
- Check for NaN values in dataset
