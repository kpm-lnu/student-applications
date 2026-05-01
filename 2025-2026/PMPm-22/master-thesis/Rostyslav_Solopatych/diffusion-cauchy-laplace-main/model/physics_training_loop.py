#!/usr/bin/env python3
"""
Train physics-informed diffusion model on a dataset.

This script trains a conditional DDPM on clean harmonic functions without corrupted samples.
The clean dataset ensures proper normalization and physically meaningful results.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import logging
import argparse

# Diffusion model components
from diffusers import UNet2DModel, DDPMScheduler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dataset components
from dataset.harmonic_field_dataset import HarmonicFieldDataset
from grid.compute_grid import create_coordinate_grids

# Physics kernels
from harmonic.kernels import get_laplacian_kernel, get_neumann_kernel
from harmonic.sample_harmonic_field import compute_normals

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_model(pixel_res=64):
    """
    Create UNet2D model for conditional diffusion.
    Hardcoded architecture with fixed 5 channels:
    - 1 channel for noisy harmonic function u
    - 4 channels for conditioning (geometry, boundary, Dirichlet, Neumann)
    
    Args:
        pixel_res: pixel resolution of input images (assumed square H=W)
    Returns:
        model: UNet2DModel instance
    """
    model = UNet2DModel(
        sample_size=pixel_res,
        in_channels=5,  # 1 (noisy u) + 4 (geometry, boundary, dirichlet, neumann)
        out_channels=1,  # predict noise on u
        layers_per_block=2,
        block_out_channels=(64, 128, 256),  # Fixed channel sizes
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D", 
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model


def create_scheduler(num_diffusion_timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Create DDPM scheduler for diffusion process:
    
    Forward process (adding noise):
        q(xₜ|x₀) = 𝒩(xₜ; √ᾱₜ x₀, (1-ᾱₜ)I) where ᾱₜ = ∏ᵢ₌₁ᵗ(1-βᵢ)
    
    Reverse process (denoising):
        pθ(xₜ₋₁|xₜ) = 𝒩(xₜ₋₁; μθ(xₜ,t), Σθ(xₜ,t)) where μθ = 1/√αₜ (xₜ - (1-αₜ)/√(1-ᾱₜ) εθ(xₜ,t))

    Args:
        num_diffusion_timesteps: number of diffusion timesteps T
        beta_start: starting noise variance β₁
        beta_end: ending noise variance βₜ
        Returns:
        scheduler: DDPMScheduler instance
    """
    scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_timesteps,
        beta_schedule="linear",
        beta_start=beta_start,
        beta_end=beta_end,
        prediction_type="epsilon"  # predict noise
    )
    return scheduler


def compute_per_sample_metrics(masked_residual, reference_tensor, mask, B):
    """
    Compute MSE and L∞ metrics per-sample, then average across batch.
    
    Args:
        masked_residual: [B, 1, H, W] residual (e.g., Laplacian, BC error) already masked
        reference_tensor: [B, 1, H, W] reference tensor for relative L∞ normalization
        mask: [B, 1, H, W] mask indicating active points
        B: batch size
    
    Returns:
        mse: scalar mean MSE across batch
        linf: scalar mean L∞ across batch
        linf_rel: scalar mean relative L∞ across batch
        mse_per_sample: [B] per-sample MSE (for timestep-dependent weighting)
    """
    # Flatten spatial dimensions for per-sample operations
    residual_flat = masked_residual.reshape(B, -1)  # [B, H*W]
    mask_flat = mask.reshape(B, -1)  # [B, H*W]
    reference_flat = reference_tensor.reshape(B, -1)  # [B, H*W]
    
    # Count active points per sample
    num_points_per_sample = mask_flat.sum(dim=1, keepdim=True) + 1e-8  # [B, 1]
    
    # Per-sample MSE
    mse_per_sample = (residual_flat ** 2).sum(dim=1, keepdim=True) / num_points_per_sample  # [B, 1]
    mse = mse_per_sample.mean()
    
    # Per-sample L∞ (max absolute error)
    linf_per_sample = torch.max(torch.abs(residual_flat), dim=1)[0]  # [B]
    linf = linf_per_sample.mean()
    
    # Per-sample reference magnitude for relative L∞
    reference_magnitude_per_sample = torch.max(torch.abs(reference_flat), dim=1)[0] + 1e-8  # [B]
    linf_rel_per_sample = linf_per_sample / reference_magnitude_per_sample  # [B]
    linf_rel = linf_rel_per_sample.mean()
    
    return mse, linf, linf_rel, mse_per_sample.squeeze(1)


def compute_physics_loss(predicted_u, cond_maps, grid_size=3.0,
                        laplacian_weight=1.0, boundary_weight=1.0, neumann_weight=1.0):
    """
    Compute physics-informed loss for harmonic functions.
    
    Args:
        predicted_u: [B, 1, H, W] predicted harmonic function
        cond_maps: [B, 4, H, W] conditioning maps (gmask, bmask, dirichlet, neumann)
        laplacian_weight: weight for the Laplace residual term
        boundary_weight: weight for the combined boundary residual term
        neumann_weight: weight for Neumann boundary loss vs Dirichlet
    
    Returns:
        physics_per_sample: [B] per-sample physics loss (combined Laplace + boundary).
            The caller is expected to multiply by a per-sample timestep weight w(t)
            and reduce via .mean() to obtain the scalar back-prop loss.
        loss_components: dict with scalar logging metrics (unweighted means / L∞).
    """
    
    B, _, H, W = predicted_u.shape
    if H <= 2 or W <= 2:
        raise ValueError("Grid size too small to compute interior Laplace loss.")

    # Extract conditioning maps
    gmask = cond_maps[:, 0:1]  # [B, 1, H, W] - geometry mask (1=interior, 0=exterior)
    bmask = cond_maps[:, 1:2]  # [B, 1, H, W] - boundary mask (1=boundary, 0=interior)

    # Grid spacing (pixel centers): same h is used for the Laplacian and the
    # central-difference gradient kernels so the residuals share dimensional scale.
    h = grid_size / W

    # Get optimized Laplacian kernel with 1/h² scaling (proper Δu approximation)
    laplacian_kernel = get_laplacian_kernel(h=h, device=predicted_u.device)

    # Apply Laplacian using F.conv2d (handles boundaries automatically with padding=1)
    laplacian_full = F.conv2d(predicted_u, laplacian_kernel, padding=1)  # [B, 1, H, W]

    # Apply geometry mask to enforce Laplace equation only in interior
    # Extract interior region (excluding boundary) to avoid edge artifacts
    gmask_interior = gmask[:, :, 1:-1, 1:-1]  # [B, 1, H-2, W-2]  
    laplacian_interior = laplacian_full[:, :, 1:-1, 1:-1]  # [B, 1, H-2, W-2]
    
    # Physics loss: Laplace equation Δu = 0 in interior (MSE + L∞ metrics)
    laplacian_masked = laplacian_interior * gmask_interior
    laplace_loss, laplace_linf, laplace_linf_rel, laplace_per_sample = compute_per_sample_metrics(
        laplacian_masked, predicted_u[:, :, 1:-1, 1:-1], gmask_interior, B
    )
    
    # Boundary condition losses
    dirichlet_bc = cond_maps[:, 2:3]  # [B, 1, H, W] - Dirichlet boundary conditions
    neumann_bc = cond_maps[:, 3:4]    # [B, 1, H, W] - Neumann boundary conditions
    
    # Dirichlet boundary condition loss: u = g on boundary (MSE + L∞ metrics)
    dirichlet_masked = (predicted_u - dirichlet_bc) * bmask
    dirichlet_loss, dirichlet_linf, dirichlet_linf_rel, dirichlet_per_sample = compute_per_sample_metrics(
        dirichlet_masked, dirichlet_bc, bmask, B
    )
    
    # Neumann boundary condition loss: ∇u·n = h on boundary
    # Reuse the same grid spacing `h` computed above for the Laplacian kernel.
    grad_x_kernel, grad_y_kernel = get_neumann_kernel(h=h, device=predicted_u.device)

    # Compute gradients using F.conv2d
    u_grad_x = F.conv2d(predicted_u, grad_x_kernel, padding=1)  # [B, 1, H, W]
    u_grad_y = F.conv2d(predicted_u, grad_y_kernel, padding=1)  # [B, 1, H, W]
    
    # For Neumann boundary conditions, we need the normal derivative ∇u·n
    # For the elliptical boundary Γ₁: x = 1.3cos(t), y = sin(t)
    X, Y = create_coordinate_grids(H, W, grid_size, device=predicted_u.device)
    B = predicted_u.shape[0]
    X = X.expand(B, -1, -1, -1)  # [B, 1, H, W]
    Y = Y.expand(B, -1, -1, -1)  # [B, 1, H, W]
    n_x, n_y = compute_normals(X, Y, device=predicted_u.device)
    normal_derivative = u_grad_x * n_x + u_grad_y * n_y
    
    # Neumann loss: match prescribed normal derivative at boundaries (MSE + L∞ metrics)
    neumann_masked = (normal_derivative - neumann_bc) * bmask
    neumann_loss, neumann_linf, neumann_linf_rel, neumann_per_sample = compute_per_sample_metrics(
        neumann_masked, neumann_bc, bmask, B
    )
    
    # Combined boundary loss with Neumann weight
    boundary_loss = dirichlet_loss + neumann_weight * neumann_loss
    
    # Total physics loss (scalar, for logging) and per-sample tensor (for w(t) weighting)
    physics_loss = laplacian_weight * laplace_loss + boundary_weight * boundary_loss
    physics_per_sample = (
        laplacian_weight * laplace_per_sample
        + boundary_weight * (dirichlet_per_sample + neumann_weight * neumann_per_sample)
    )  # [B]
    
    loss_components = {
        'laplace': laplace_loss.item(),
        'dirichlet': dirichlet_loss.item(),
        'neumann': neumann_loss.item(),
        'neumann_scaled': (neumann_weight * neumann_loss).item(),
        'boundary': boundary_loss.item(),
        'total_physics': physics_loss.item(),
        # L∞ metrics (absolute) for interpretability
        'laplace_linf': laplace_linf.item(),
        'dirichlet_linf': dirichlet_linf.item(),
        'neumann_linf': neumann_linf.item(),
        # L∞ metrics (relative) for scale-invariant comparison
        'laplace_linf_rel': laplace_linf_rel.item(),
        'dirichlet_linf_rel': dirichlet_linf_rel.item(),
        'neumann_linf_rel': neumann_linf_rel.item(),
    }
    
    return physics_per_sample, loss_components


def compute_batch_losses(u_batch, cond_batch, model, scheduler, device,
                         grid_size, laplacian_weight, boundary_weight, neumann_weight):
    """
    Compute diffusion and physics losses for a batch.

    The physics term follows the PIDM formulation (Bastek et al. 2024,
    arXiv:2403.14404, ICLR 2025): the per-sample physics residual R(x̂₀)
    is treated as a Gaussian observation with variance equal to the DDPM
    posterior variance Σ_t (with the convention Σ_0 := Σ_1 to avoid 0/0
    at t=0). The per-sample physics weight is therefore

        w(t) = 1 / (2 · Σ_t),
        Σ_t  = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) · β_t,
        Σ_0 := Σ_1.

    so high-noise timesteps (Σ_t ≈ β_T) downweight the physics loss while
    low-noise timesteps (small Σ_t) upweight it sharply.

        L = L_simple + E_{t,x₀,ε}[ (1 / (2 Σ_t)) · ||R(x̂₀(x_t,t))||² ]

    Args:
        u_batch: [B, 1, H, W] batch of harmonic functions
        cond_batch: [B, 4, H, W] batch of conditioning maps
        model: UNet2D model
        scheduler: DDPM scheduler
        device: torch device
        grid_size: physical domain size
        laplacian_weight: weight for the Laplace residual term
        boundary_weight: weight for the combined boundary residual term
        neumann_weight: weight for Neumann boundary loss vs Dirichlet

    Returns:
        diffusion_loss: scalar diffusion loss
        physics_loss: scalar w(t)-weighted physics loss (back-prop quantity)
        total_loss: scalar total loss
        physics_components: dict with individual (unweighted) physics metrics
        mean_w_t: scalar mean of w(t)=1/(2Σ_t) across the batch (for logging)
    """
    # Sample random timesteps
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps,
        (u_batch.shape[0],), device=device
    ).long()

    # Add gaussian noise to clean "images"
    noise = torch.randn_like(u_batch)
    noisy_u = scheduler.add_noise(u_batch, noise, timesteps)

    # Concatenate noisy u with conditioning
    model_input = torch.cat([noisy_u, cond_batch], dim=1)  # [B, 5, H, W]

    # Predict noise
    noise_pred = model(model_input, timesteps).sample

    # Diffusion loss (MSE between predicted and actual noise)
    diffusion_loss = F.mse_loss(noise_pred, noise)

    # Physics is disabled when ALL of laplacian_weight, boundary_weight, neumann_weight are 0;
    # in that case the entire physics block (inversion + residuals) is skipped
    physics_enabled = (
        (laplacian_weight > 0.0) or (boundary_weight > 0.0) or (neumann_weight > 0.0)
    )

    # PIDM timestep weighting: w(t) = 1 / (2 Σ_t), with
    #   Σ_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) · β_t
    # and the convention Σ_0 := Σ_1 (clip at t=0 to avoid division by zero).
    # Computed even when physics is disabled so it can be logged.
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    betas = scheduler.betas.to(device)
    alpha_bar_t = alphas_cumprod[timesteps]                              # [B]
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, device=device, dtype=alphas_cumprod.dtype),
         alphas_cumprod[:-1]]
    )                                                                   # [T]
    posterior_variance_full = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # Clip Σ_0 := Σ_1
    posterior_variance_clipped = posterior_variance_full.clone()
    posterior_variance_clipped[0] = posterior_variance_full[1]
    sigma_t = posterior_variance_clipped[timesteps]                     # [B]
    w_t_per_sample = 0.5 / sigma_t                                      # [B]
    mean_w_t = w_t_per_sample.mean().item()

    # Reshape ᾱ_t for the x̂₀ inversion broadcast
    alpha_t = alpha_bar_t.view(-1, 1, 1, 1)

    if physics_enabled:
        # Physics loss on recovered-denoised prediction
        # DDPM denoising formula: x̂₀ = (x_t - √(1-ᾱ_t) * ε̂) / √ᾱ_t
        predicted_clean_u = (noisy_u - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

        # Per-sample physics residual [B] and scalar logging metrics
        physics_per_sample, physics_components = compute_physics_loss(
            predicted_clean_u, cond_batch,
            grid_size=grid_size,
            laplacian_weight=laplacian_weight,
            boundary_weight=boundary_weight,
            neumann_weight=neumann_weight,
        )

        # PIDM timestep weighting: w(t) = 1 / (2 Σ_t) (per-sample, raw, no normalization)
        physics_loss = (w_t_per_sample * physics_per_sample).mean()
    else:
        physics_loss = torch.tensor(0.0, device=device)
        physics_components = {
            'laplace': 0.0,
            'dirichlet': 0.0,
            'neumann': 0.0,
            'neumann_scaled': 0.0,
            'boundary': 0.0,
            'total_physics': 0.0,
            'laplace_linf': 0.0,
            'dirichlet_linf': 0.0,
            'neumann_linf': 0.0,
            'laplace_linf_rel': 0.0,
            'dirichlet_linf_rel': 0.0,
            'neumann_linf_rel': 0.0,
        }

    # Combined loss
    total_loss = diffusion_loss + physics_loss

    return diffusion_loss, physics_loss, total_loss, physics_components, mean_w_t


def run_epoch(model, dataloader, scheduler, device,
              grid_size, laplacian_weight, boundary_weight, neumann_weight, optimizer=None,
              epoch_num=None, total_epochs=None, phase='Train'):
    """
    Run one epoch of training or validation.
    
    Args:
        model: UNet2D model
        dataloader: DataLoader for training or validation
        scheduler: DDPM scheduler
        device: torch device
        grid_size: physical domain size
        boundary_weight: weight for boundary loss
        neumann_weight: weight for Neumann boundary loss
        optimizer: optimizer for training (None for validation)
        epoch_num: current epoch number (for progress bar)
        total_epochs: total number of epochs (for progress bar)
        phase: 'Train' or 'Val' for progress bar description
    
    Returns:
        avg_diffusion_loss: average diffusion loss for the epoch
        avg_physics_loss: average physics loss for the epoch
        avg_total_loss: average total loss for the epoch
        avg_physics_components: averaged physics components across all batches
        avg_mean_w_t: average per-batch mean of w(t)=1/(2Σ_t) (for logging only)
    """
    is_training = optimizer is not None
    
    if is_training:
        model.train()
    else:
        model.eval()
    
    epoch_diffusion_loss = 0.0
    epoch_physics_loss = 0.0
    epoch_total_loss = 0.0
    epoch_boundary_loss = 0.0
    epoch_mean_w_t = 0.0
    
    # Track physics components across all batches
    epoch_laplace_loss = 0.0
    epoch_dirichlet_loss = 0.0
    epoch_neumann_loss = 0.0
    epoch_neumann_scaled_loss = 0.0
    
    # Track L∞ metrics (absolute)
    epoch_laplace_linf = 0.0
    epoch_dirichlet_linf = 0.0
    epoch_neumann_linf = 0.0
    
    # Track L∞ metrics (relative)
    epoch_laplace_linf_rel = 0.0
    epoch_dirichlet_linf_rel = 0.0
    epoch_neumann_linf_rel = 0.0
    
    # Create progress bar
    desc = f"Epoch {epoch_num}/{total_epochs} [{phase}]" if epoch_num else f"[{phase}]"
    progress_bar = tqdm(dataloader, desc=desc)
    
    # Context manager for validation (no gradients)
    context = torch.enable_grad() if is_training else torch.no_grad()
    
    with context:
        for u_batch, cond_batch in progress_bar:
            u_batch = u_batch.to(device)
            cond_batch = cond_batch.to(device)
            
            # Compute losses
            diffusion_loss, physics_loss, total_loss, physics_components, mean_w_t = compute_batch_losses(
                u_batch, cond_batch, model, scheduler, device,
                grid_size, laplacian_weight, boundary_weight, neumann_weight
            )
            
            # Backward pass (only for training)
            if is_training:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Accumulate losses
            epoch_diffusion_loss += diffusion_loss.item()
            epoch_physics_loss += physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss
            epoch_total_loss += total_loss.item()
            epoch_mean_w_t += mean_w_t
            
            # Accumulate physics components
            epoch_laplace_loss += physics_components['laplace']
            epoch_dirichlet_loss += physics_components['dirichlet']
            epoch_neumann_loss += physics_components['neumann']
            epoch_neumann_scaled_loss += physics_components['neumann_scaled']
            epoch_boundary_loss += physics_components['boundary']
            epoch_laplace_linf += physics_components['laplace_linf']
            epoch_dirichlet_linf += physics_components['dirichlet_linf']
            epoch_neumann_linf += physics_components['neumann_linf']
            epoch_laplace_linf_rel += physics_components['laplace_linf_rel']
            epoch_dirichlet_linf_rel += physics_components['dirichlet_linf_rel']
            epoch_neumann_linf_rel += physics_components['neumann_linf_rel']
            
            # Update progress bar (show MSE and relative L∞ for Laplace)
            if is_training:
                progress_bar.set_postfix({
                    'diffusion': f"{diffusion_loss.item():.4f}",
                    'MSE Laplace': f"{physics_components['laplace']:.4f}",
                    'L∞ Laplace': f"{physics_components['laplace_linf']:.4f}",
                    'Rel L∞ Laplace': f"{physics_components['laplace_linf_rel']:.4f}",
                    'total': f"{total_loss.item():.4f}",
                    'mean_w_t': f"{mean_w_t:.3f}"
                })
            else:
                progress_bar.set_postfix({
                    'val_diff': f"{diffusion_loss.item():.4f}",
                    'val_total': f"{total_loss.item():.4f}"
                })
    
    # Compute averages
    num_batches = len(dataloader)
    avg_diffusion_loss = epoch_diffusion_loss / num_batches
    avg_physics_loss = epoch_physics_loss / num_batches
    avg_total_loss = epoch_total_loss / num_batches
    avg_mean_w_t = epoch_mean_w_t / num_batches
    
    # Compute averaged physics components (including L∞ metrics)
    avg_physics_components = {
        'laplace': epoch_laplace_loss / num_batches,
        'dirichlet': epoch_dirichlet_loss / num_batches,
        'neumann': epoch_neumann_loss / num_batches,
        'neumann_scaled': epoch_neumann_scaled_loss / num_batches,
        'boundary': epoch_boundary_loss / num_batches,
        'total_physics': avg_physics_loss,
        # L∞ metrics (absolute, averaged across batches)
        'laplace_linf': epoch_laplace_linf / num_batches,
        'dirichlet_linf': epoch_dirichlet_linf / num_batches,
        'neumann_linf': epoch_neumann_linf / num_batches,
        # L∞ metrics (relative, averaged across batches)
        'laplace_linf_rel': epoch_laplace_linf_rel / num_batches,
        'dirichlet_linf_rel': epoch_dirichlet_linf_rel / num_batches,
        'neumann_linf_rel': epoch_neumann_linf_rel / num_batches,
    }
    
    return avg_diffusion_loss, avg_physics_loss, avg_total_loss, avg_physics_components, avg_mean_w_t


def train_model(dataset_file, num_epochs=20, pixel_res=64, batch_size=16, learning_rate=1e-4,
                num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                grid_size=3.0, laplacian_weight=0.0, boundary_weight=0.0, neumann_weight=0.0,
                save_dir="./checkpoints", save_every=5, resume_from_checkpoint=None, device=None,
                val_dataset_file=None, early_stopping_patience=None, early_stopping_metric='physics'):
    """
    Train the diffusion model on dataset.
    
    Args:
        dataset_file: path to training dataset
        val_dataset_file: path to validation dataset (optional)
        num_epochs: number of training epochs
        pixel_res: grid size H×W for model
        batch_size: training batch size
        learning_rate: learning rate
        num_diffusion_timesteps: number of diffusion timesteps
        beta_start: starting noise variance
        beta_end: ending noise variance
        grid_size: physical domain size (assumed square)
        laplacian_weight: weight for the Laplace residual term
        boundary_weight: weight for boundary loss
        neumann_weight: weight for Neumann boundary loss
        save_dir: directory to save checkpoints
        save_every: save checkpoint every N epochs (rolling best checkpoint is always
            kept separately under <save_dir>/cond_ddpm_best/ when validation data is
            provided, regardless of save_every)
        resume_from_checkpoint: path to checkpoint to resume from
        early_stopping_patience: number of epochs without improvement before stopping (optional, None to disable)
        early_stopping_metric: metric to use for early stopping AND for selecting the
            rolling best checkpoint - 'physics' or 'diffusion' (default: 'physics')
        device: torch device to use (cpu or cuda)
    
    Returns:
        model: trained model
        scheduler: scheduler
        epoch_losses: list of dicts with losses for each epoch
    """
    logger.info(f"Training on device: {device}")

    # Load train dataset and metadata
    train_dataset = torch.load(dataset_file, weights_only=False)
    
    # Try to load metadata from the base filename
    if '_train.pt' in dataset_file:
        metadata_file = dataset_file.replace('_train.pt', '_metadata.pt')
    else:
        metadata_file = dataset_file.replace('.pt', '_metadata.pt')
    
    metadata = torch.load(metadata_file, weights_only=False)

    logger.info(f"Loaded training dataset: {len(train_dataset)} samples")
    logger.info(f"Normalization stats: μ={metadata['global_mu']:.6f}, σ={metadata['global_sigma']:.6f}")
    
    # Create train dataset wrapper
    train_harmonic_dataset = HarmonicFieldDataset(
        data_list=train_dataset,
        mu_u=metadata['global_mu'],
        sigma_u=metadata['global_sigma']
    )
    
    # Create train data loader
    train_dataloader = DataLoader(
        train_harmonic_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    logger.info(f"Train data loader created: {len(train_dataloader)} batches per epoch")
    
    # Load validation dataset if provided
    val_dataloader = None
    if val_dataset_file and os.path.exists(val_dataset_file):
        val_dataset = torch.load(val_dataset_file, weights_only=False)
        logger.info(f"Loaded validation dataset: {len(val_dataset)} samples")
        
        val_harmonic_dataset = HarmonicFieldDataset(
            data_list=val_dataset,
            mu_u=metadata['global_mu'],
            sigma_u=metadata['global_sigma']
        )
        
        val_dataloader = DataLoader(
            val_harmonic_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=0,
            drop_last=False
        )
        
        logger.info(f"Validation data loader created: {len(val_dataloader)} batches")
    else:
        logger.info("No validation dataset provided or file not found")
    
    # Create model and scheduler
    model = create_model(pixel_res).to(device)
    scheduler = create_scheduler(
        num_diffusion_timesteps=num_diffusion_timesteps,
        beta_start=beta_start,
        beta_end=beta_end
    )
    
    # Resume from checkpoint if specified
    if resume_from_checkpoint:
        if os.path.exists(resume_from_checkpoint):
            logger.info(f"Loading checkpoint from: {resume_from_checkpoint}")
            model = UNet2DModel.from_pretrained(resume_from_checkpoint).to(device)
            logger.info(f"Resumed from checkpoint: {resume_from_checkpoint}")
        else:
            logger.warning(f"Checkpoint not found: {resume_from_checkpoint}")
            logger.info("Starting training from scratch...")
    
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    os.makedirs(save_dir, exist_ok=True)
    
    model.train()
    
    # Track epoch losses for returning/outputting
    epoch_losses = []
    
    # Determine starting epoch from checkpoint
    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        try:
            # Extract epoch number from checkpoint path like "cond_ddpm_epoch15"
            checkpoint_name = os.path.basename(resume_from_checkpoint.rstrip('/'))
            start_epoch = int(checkpoint_name.split('epoch')[-1])
            logger.info(f"Resuming training from epoch {start_epoch}")
        except (ValueError, IndexError):
            logger.warning(f"Could not extract epoch number from {resume_from_checkpoint}, starting from epoch 0")
            start_epoch = 0
    
    # Best-checkpoint / early-stopping tracking
    best_val_loss = float('inf')
    best_epoch = None
    epochs_without_improvement = 0
    early_stopped = False
    best_checkpoint_dir = os.path.join(save_dir, "cond_ddpm_best")
    best_scheduler_dir = os.path.join(save_dir, "scheduler_best")
    best_meta_path = os.path.join(save_dir, "best_checkpoint.json")
    
    # Training loop with correct epoch range
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Training phase
        (avg_train_diffusion_loss, avg_train_physics_loss, avg_train_total_loss,
         avg_train_physics_components, avg_train_mean_w_t) = run_epoch(
            model=model,
            dataloader=train_dataloader,
            scheduler=scheduler,
            device=device,
            grid_size=grid_size,
            laplacian_weight=laplacian_weight,
            boundary_weight=boundary_weight,
            neumann_weight=neumann_weight,
            optimizer=optimizer,
            epoch_num=epoch + 1,
            total_epochs=start_epoch + num_epochs,
            phase='Train'
        )
        
        # Validation phase
        val_results = None
        if val_dataloader is not None:
            (avg_val_diffusion_loss, avg_val_physics_loss, avg_val_total_loss,
             _, avg_val_mean_w_t) = run_epoch(
                model=model,
                dataloader=val_dataloader,
                scheduler=scheduler,
                device=device,
                grid_size=grid_size,
                laplacian_weight=laplacian_weight,
                boundary_weight=boundary_weight,
                neumann_weight=neumann_weight,
                optimizer=None,  # No optimizer for validation
                epoch_num=epoch + 1,
                total_epochs=start_epoch + num_epochs,
                phase='Val'
            )
            
            val_results = {
                'diffusion': avg_val_diffusion_loss,
                'physics': avg_val_physics_loss,
                'total': avg_val_total_loss,
                'mean_w_t': avg_val_mean_w_t,
            }
        
        # Epoch summary with train and validation results
        logger.info(f"Epoch {epoch+1}/{start_epoch + num_epochs} completed:")
        logger.info(f"  [TRAIN] Diffusion Loss MSE: {avg_train_diffusion_loss:.6f}")
        logger.info(f"  [TRAIN] Physics Loss (w(t)-weighted, 1+2+4): {avg_train_physics_loss:.6f}")
        logger.info(f"    1.  Laplace Loss MSE: {avg_train_physics_components['laplace']:.6f}")
        logger.info(f"    2.  Dirichlet Loss MSE: {avg_train_physics_components['dirichlet']:.6f}")
        logger.info(f"    3.  Neumann Loss MSE: {avg_train_physics_components['neumann']:.6f}")
        logger.info(f"    4.  Neumann (scaled) Loss MSE: {avg_train_physics_components['neumann_scaled']:.6f}")
        logger.info(f"    5.  Laplace Loss L∞: {avg_train_physics_components['laplace_linf']:.6f}")
        logger.info(f"    6.  Dirichlet L∞: {avg_train_physics_components['dirichlet_linf']:.6f}")
        logger.info(f"    7.  Neumann L∞: {avg_train_physics_components['neumann_linf']:.6f}")
        logger.info(f"    8.  Rel. Laplace Loss L∞: {avg_train_physics_components['laplace_linf_rel']:.6f}")
        logger.info(f"    9.  Rel. Dirichlet L∞: {avg_train_physics_components['dirichlet_linf_rel']:.6f}")
        logger.info(f"    10. Rel. Neumann L∞: {avg_train_physics_components['neumann_linf_rel']:.6f}")
        
        logger.info(f"  [TRAIN] Total Loss MSE: {avg_train_total_loss:.6f}")
        
        if val_results is not None:
            logger.info(f"  [VAL] Diffusion Loss MSE: {val_results['diffusion']:.6f}")
            logger.info(f"  [VAL] Physics Loss MSE: {val_results['physics']:.6f}")
            logger.info(f"  [VAL] Total Loss MSE: {val_results['total']:.6f}")
        
        logger.info(f"  Mean w(t) = 1/(2Σ_t) (PIDM, train): {avg_train_mean_w_t:.4f}")
        
        # Best-checkpoint tracking (always when validation data is available)
        if val_results is not None:
            if early_stopping_metric == 'physics':
                current_val_loss = val_results['physics']
                metric_name = 'physics'
            elif early_stopping_metric == 'diffusion':
                current_val_loss = val_results['diffusion']
                metric_name = 'diffusion'
            else:
                raise ValueError(f"Invalid early_stopping_metric: {early_stopping_metric}. Must be 'physics' or 'diffusion'.")

            improved = current_val_loss < best_val_loss
            if improved:
                best_val_loss = current_val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                logger.info(f"  ✓ New best validation {metric_name} loss: {best_val_loss:.6f}")

                # Persist rolling best checkpoint (overwrites previous best)
                model.save_pretrained(best_checkpoint_dir)
                os.makedirs(best_scheduler_dir, exist_ok=True)
                with open(os.path.join(best_scheduler_dir, "scheduler_config.json"), 'w') as f:
                    json.dump(scheduler.config, f, indent=2)
                with open(best_meta_path, 'w') as f:
                    json.dump({
                        'epoch': best_epoch,
                        'metric_name': metric_name,
                        'val_loss': best_val_loss,
                    }, f, indent=2)
                logger.info(f"  ✓ Saved best checkpoint: {best_checkpoint_dir}")
            else:
                epochs_without_improvement += 1
                logger.info(f"  No improvement for {epochs_without_improvement} epoch(s)")

            # Early stopping check (only if patience is configured)
            if early_stopping_patience is not None and not improved:
                if epochs_without_improvement >= early_stopping_patience:
                    current_epoch = epoch + 1
                    logger.info(f"\n⏹ Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                    logger.info(f"  ✓ Best epoch: {best_epoch} with val_{metric_name}_loss={best_val_loss:.6f} (early stopped at epoch {current_epoch})")
                    early_stopped = True
        
        # Store epoch results
        epoch_result = {
            'epoch': epoch + 1,
            'train_diffusion_loss_mse': avg_train_diffusion_loss,
            'train_physics_loss_mse': avg_train_physics_loss,
            'train_total_loss_mse': avg_train_total_loss,
            'mean_w_t': avg_train_mean_w_t,
            'laplace_loss_mse': avg_train_physics_components['laplace'],
            'dirichlet_loss_mse': avg_train_physics_components['dirichlet'],
            'neumann_loss_mse': avg_train_physics_components['neumann'],
            'neumann_scaled_loss_mse': avg_train_physics_components['neumann_scaled'],
            'laplace_loss_linf': avg_train_physics_components['laplace_linf'],
            'dirichlet_loss_linf': avg_train_physics_components['dirichlet_linf'],
            'neumann_loss_linf': avg_train_physics_components['neumann_linf'],
        }
        if val_results is not None:
            epoch_result['val_diffusion_loss_mse'] = val_results['diffusion']
            epoch_result['val_physics_loss_mse'] = val_results['physics']
            epoch_result['val_total_loss_mse'] = val_results['total']
        epoch_losses.append(epoch_result)
        
        # Save epoch losses after each epoch (for recovery in case of interruption)
        training_epoch_output_json = os.path.join(save_dir, "training_epoch_losses.json")
        with open(training_epoch_output_json, 'w') as f:
            json.dump({'epoch_losses': epoch_losses}, f, indent=2)

        # Save checkpoint every N epochs
        if (epoch + 1) % save_every == 0:
            checkpoint_dir = os.path.join(save_dir, f"cond_ddpm_epoch{epoch+1}")
            model.save_pretrained(checkpoint_dir)
            
            # Save scheduler config
            scheduler_dir = os.path.join(save_dir, f"scheduler_epoch{epoch+1}.json")
            os.makedirs(scheduler_dir, exist_ok=True)
            scheduler_config_path = os.path.join(scheduler_dir, "scheduler_config.json")
            with open(scheduler_config_path, 'w') as f:
                json.dump(scheduler.config, f, indent=2)
            
            logger.info(f"Saved checkpoint: {checkpoint_dir}")
        
        # Break if early stopping triggered
        if early_stopped:
            break
    
    logger.info(f"Training completed! Epoch losses saved to: {os.path.join(save_dir, 'training_epoch_losses.json')}")
    if best_epoch is not None:
        logger.info(f"Best checkpoint: {best_checkpoint_dir} (epoch {best_epoch}, val_{early_stopping_metric}_loss={best_val_loss:.6f})")
    return model, scheduler, epoch_losses


def main():
    """Main training function with command-line argument support."""
    
    parser = argparse.ArgumentParser(description='Train physics-informed diffusion model on harmonic functions')
    
    # Dataset and I/O arguments
    parser.add_argument('--dataset', type=str, default='harmonic_field_dataset_train.pt',
                        help='Path to training dataset file (default: harmonic_field_dataset_train.pt)')
    parser.add_argument('--val_dataset', type=str, default=None,
                        help='Path to validation dataset file (default: None, auto-detected from --dataset)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for checkpoints (default: ./checkpoints)')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint directory to resume training from (default: None)')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Number of epochs without improvement before early stopping (default: None, disabled)')
    parser.add_argument('--early_stopping_metric', type=str, default='physics',
                        choices=['physics', 'diffusion'],
                        help='Metric to use for early stopping: physics (L2 error) or diffusion (MSE) (default: physics)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training (default: auto)')
    
    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimizer (default: 1e-4)')
    
    # Model configuration
    parser.add_argument('--num_diffusion_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps (default: 1000)')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='Starting noise variance β₁ (default: 0.0001)')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='Ending noise variance βₜ (default: 0.02)')
    parser.add_argument('--pixel_res', type=int, default=64,
                        help='Grid size H×W for model (default: 64)')
    parser.add_argument('--grid_size', type=float, default=3.0,
                        help='Physical grid size (default: 3.0 for domain [-1.5,1.5]x[-1.5,1.5])')
    
    # Physics loss weights. All three default to 0.0, which DISABLES the physics term entirely (inversion + residuals are skipped).
    # Pass nonzero values explicitly to enable PIDM training, e.g. --laplacian_weight 1.0 --boundary_weight 1.0 --neumann_weight 1.0
    parser.add_argument('--laplacian_weight', type=float, default=0.0,
                        help='Weight for the Laplace residual term. Default 0.0 disables physics.')
    parser.add_argument('--boundary_weight', type=float, default=0.0,
                        help='Weight for boundary condition losses. Default 0.0 disables physics.')
    parser.add_argument('--neumann_weight', type=float, default=0.0,
                        help='Weight for Neumann boundary condition losses. Default 0.0 disables physics.')

    args = parser.parse_args()
    
    # Check dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        logger.error("Please ensure the dataset exists and is clean")
        return
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Auto-detect validation dataset if not provided
    val_dataset_file = args.val_dataset
    if val_dataset_file is None:
        # Try to auto-detect validation file
        if '_train.pt' in args.dataset:
            val_dataset_file = args.dataset.replace('_train.pt', '_val.pt')
            if os.path.exists(val_dataset_file):
                logger.info(f"Auto-detected validation dataset: {val_dataset_file}")
            else:
                val_dataset_file = None
                logger.info("No validation dataset found")
    
    # Training configuration from arguments
    config = {
        'dataset_file': args.dataset,
        'val_dataset_file': val_dataset_file,
        'num_epochs': args.num_epochs,
        'pixel_res': args.pixel_res,
        'grid_size': args.grid_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_diffusion_timesteps': args.num_diffusion_timesteps,
        'beta_start': args.beta_start,
        'beta_end': args.beta_end,
        'laplacian_weight': args.laplacian_weight,
        'boundary_weight': args.boundary_weight,
        'neumann_weight': args.neumann_weight,
        'save_dir': args.output_dir,
        'save_every': args.save_every,
        'resume_from_checkpoint': args.resume_from_checkpoint,
        'early_stopping_patience': args.early_stopping_patience,
        'early_stopping_metric': args.early_stopping_metric,
        'device': device,
    }
    
    logger.info("Starting training with physics-informed diffusion model...")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Configuration: {config}")

    # Train model
    model, scheduler, epoch_losses = train_model(**config)
    
    logger.info("✅ Training completed successfully!")
    logger.info("✅ Model trained on clean dataset with proper normalization")
    logger.info(f"✅ Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()