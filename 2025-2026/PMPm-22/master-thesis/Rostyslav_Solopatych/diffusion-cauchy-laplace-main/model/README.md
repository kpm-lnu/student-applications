# Model Training and Sample Generation

This directory contains scripts for training physics-informed diffusion model based (PIDM) based on Denoising Diffusion Probabilistic Model (DDPM) to solve the Cauchy problem for Laplace equation on a bounded planar domain with 1 hole.

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
- Early Stopping: Optional patience-based early stopping on validation loss with configurable metric (physics MSE error or diffusion MSE loss, disabled by default)
- Best Checkpoint Tracking: Whenever validation data is provided, the model with the lowest validation `--early_stopping_metric` so far is kept as a rolling `cond_ddpm_best/` checkpoint (overwritten on each new best). This is independent of `--save_every`, so a large `--save_every` can be used to save disk space without losing the best parameters.
- Structured output: Outputs epoch-level training/validation losses to JSON file
- In-memory tracking: Returns all epoch losses for programmatic access

### 2. `grid_search_hyperparameters.py`

Automated hyperparameter optimization script that performs comprehensive grid search to find the best training configuration:

- **Search Space**:
  - `batch_size`: [32, 64, 128]
  - `learning_rate`: [1e-3, 1e-4, 1e-5]
  - `pixel_res`: [64, 128]
  - `num_epochs`: 100 (automatically selects best checkpoint)
- **Smart Dataset Management**: Regenerates datasets for each `pixel_res`
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
- **Other**: `mean_w_t`

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

### Training a Model

#### Basic Training

Train for 120 epochs with default parameters (automatically includes validation):

```bash
python model/physics_training_loop.py --output_dir ./checkpoints_stage_1 \
  --dataset ./harmonic_field_100k_train.pt \
  --val_dataset ./harmonic_field_100k_val.pt \
  --save_every 1 \
  --early_stopping_patience 10 \
  --early_stopping_metric physics \
  --num_epochs 120 \
  --batch_size 128 \
  --learning_rate 1e-3 \
  --num_diffusion_timesteps 1000 \
  --beta_start 0.0001 \
  --beta_end 0.02 \
  --pixel_res 64 \
  --grid_size 3.0 \
  --laplacian_weight 1.0 \
  --boundary_weight 1.0 \
  --neumann_weight 1.0 \
  --device cuda
```

**Note**: The validation dataset is auto-detected (looks for `*_val.pt` matching the training file). You can explicitly specify it with `--val_dataset` if needed.

#### Parameters Explained

**Dataset & I/O:**

- `--output_dir`: Directory to save model checkpoints (default: `./checkpoints`)
- `--dataset`: Path to training dataset file (default: `harmonic_field_dataset_train.pt`)
- `--val_dataset`: Path to validation dataset file (default: auto-detected from training file)
- `--save_every`: Save checkpoint every N epochs (default: `5`). The rolling best checkpoint at `cond_ddpm_best/` is updated independently whenever validation data is provided.
- `--resume_from_checkpoint`: Path to checkpoint directory to resume from (default: `None`)
- `--device`: Device for training - `auto`, `cuda`, or `cpu` (default: `auto`)

**Training Hyperparameters:**

- `--num_epochs`: Number of training epochs (default: `5`)
- `--batch_size`: Training batch size (default: `16`)
- `--learning_rate`: Adam optimizer learning rate (default: `1e-4`)
- `--early_stopping_patience`: Number of epochs without improvement before stopping (default: `None`, disabled)
- `--early_stopping_metric`: Metric to use for early stopping AND for selecting the rolling best checkpoint - `physics` (L2 error) or `diffusion` (MSE) (default: `physics`)

**Model Configuration:**

- `--pixel_res`: Grid resolution H×W (default: `64`)
- `--grid_size`: Physical domain size, e.g., 3.0 for [-1.5,1.5]×[-1.5,1.5] (default: `3.0`)
- `--num_diffusion_timesteps`: Number of diffusion timesteps T (default: `1000`)
- `--beta_start`: Starting noise variance β₁ (default: `0.0001`)
- `--beta_end`: Ending noise variance βₜ (default: `0.02`)

**Physics Loss Weights:**

- `--laplacian_weight`: Weight for the Laplace residual term (default: `0.0`). **Set nonzero (e.g. `1.0`) to enable physics.**
- `--boundary_weight`: Weight for the combined boundary residual term (default: `0.0`). **Set nonzero (e.g. `1.0`) to enable physics.**
- `--neumann_weight`: Relative weight for Neumann vs Dirichlet losses (default: `0.0`). **Set nonzero (e.g. `1.0`) to enable physics.**

The combined per-sample residual is `R = laplacian_weight · R_laplace + boundary_weight · (R_dirichlet + neumann_weight · R_neumann)`.

**Physics is disabled by default.** When `--laplacian_weight`, `--boundary_weight`, and `--neumann_weight` are all `0.0`, the entire physics block (inversion + residual computation) is skipped, leaving pure DDPM training. Pass nonzero values explicitly to enable PIDM training.

**Note on PIDM timestep weighting:** Following Bastek et al. 2024 (arXiv:2403.14404, ICLR 2025), the physics term is weighted per-sample by `w(t) = 1 / (2 Σ_t)` where `Σ_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) · β_t` is the DDPM reverse-process posterior variance. The weight comes from treating the residual `R(x̂₀)` as a Gaussian observation with variance `Σ_t` (the negative-log-likelihood is `R² / (2 Σ_t)`). The convention `Σ_0 := Σ_1` is used to avoid a `0/0` at `t=0`, matching the reference PIDM implementation. There is no outer scalar `physics_weight` and no epoch curriculum; the per-batch mean of `w(t)` is logged each epoch as `mean_w_t`.

**Note on Physics Loss Computation:** The physics losses (Laplace, Dirichlet, and Neumann) are computed using **MSE (Mean Squared Error)** for training, which provides smoother gradients and better convergence. Additionally, **L∞ (max absolute error) metrics** are logged for interpretability, showing the worst-case pointwise violation of physical constraints.

#### Resume Training from Checkpoint

To continue training from epoch 15:

```bash
python model/physics_training_loop.py --output_dir ./checkpoints \
    --dataset harmonic_field_dataset_train.pt \
    --num_epochs 10 \
    --resume_from_checkpoint ./checkpoints/cond_ddpm_epoch15
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
checkpoint_path = "./checkpoints/cond_ddpm_best"
scheduler_config_path = "./checkpoints/scheduler_best/scheduler_config.json"

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
  - 1 channel: initial noise to be transformed into harmonic function `u(x,y)`
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

Following PIDM (Bastek et al. 2024, [arXiv:2403.14404](https://arxiv.org/abs/2403.14404), ICLR 2025):

```
L_total = L_diffusion + E_{t,x₀,ε}[ (1 / (2 Σ_t)) · L_physics(x̂₀(x_t,t)) ]
where Σ_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) · β_t  and  Σ_0 := Σ_1
```

Where:

- **L_diffusion**: MSE between predicted and actual noise (`L_simple` in DDPM)
- **x̂₀**: recovered clean estimate, x̂₀ = (x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t
- **w(t) = 1 / (2 Σ_t)**: per-sample timestep weight derived from the Gaussian-residual interpretation of the physics term in PIDM (the residual `R(x̂₀)` is treated as having posterior variance `Σ_t`, so its negative log-likelihood is `R² / (2 Σ_t)`). High-noise timesteps (large `Σ_t`) downweight the physics term since `x̂₀` is unreliable there; low-noise timesteps (small `Σ_t`) upweight it.
- **L_physics**: per-sample residual evaluated on x̂₀, combined as:
  - **L_laplace**: MSE of ∇²u in interior (should be ≈ 0)
  - **L_dirichlet**: MSE of (u - g) at boundary points
  - **L_neumann**: MSE of (∇u·n - h) at boundary points
  - **L_boundary** = L_dirichlet + λ_neumann \* L_neumann
  - **L_physics** = L_laplace + λ_boundary \* L_boundary

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
├── cond_ddpm_best/                # Rolling best checkpoint (lowest val early_stopping_metric)
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
│   └── diffusion_pytorch_model.safetensors
├── scheduler_best/
│   └── scheduler_config.json
├── best_checkpoint.json           # {epoch, metric_name, val_loss} for the rolling best
├── cond_ddpm_epoch5/              # Periodic snapshots controlled by --save_every
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

The rolling best checkpoint is only written when validation data is provided. With a large `--save_every`, the periodic `cond_ddpm_epoch*` directories may be sparse or absent, but `cond_ddpm_best/` is always kept current.

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
