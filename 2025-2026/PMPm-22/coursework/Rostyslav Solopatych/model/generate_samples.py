import torch
import sys
import os
from tqdm import tqdm
import json
from diffusers import UNet2DModel, DDPMScheduler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_trained_model(checkpoint_path, scheduler_config_path, device):
    """
    Load trained model and scheduler from checkpoints.
    
    Parameters:
    - checkpoint_path: path to model checkpoint directory
    - scheduler_config_path: path to scheduler config JSON file
    - device: torch device
    
    Returns:
    - model: loaded UNet2DModel
    - scheduler: loaded DDPMScheduler
    """
    print(f"Loading model from: {checkpoint_path}")
    model = UNet2DModel.from_pretrained(checkpoint_path).to(device)
    model.eval()
    
    print(f"Loading scheduler config from: {scheduler_config_path}")
    with open(scheduler_config_path, 'r') as f:
        scheduler_config = json.load(f)
    
    scheduler = DDPMScheduler(**scheduler_config)
    
    print(f"Model loaded successfully on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, scheduler


def generate_samples(model, scheduler, cond, mu, sigma, denoising_steps=1000, device=None):
    """
    Generate samples using the trained diffusion model.
    
    Parameters:
    - model: trained UNet2DModel
    - scheduler: DDPMScheduler
    - cond: conditioning tensor of shape [B,4,H,W] (geom, bmask, d_norm, n_norm)
    - mu: mean used for normalization during training
    - sigma: std used for normalization during training
    - denoising_steps: number of denoising steps
    - device: torch device
    
    Returns:
    - samples: generated samples of shape [B,1,H,W] (denormalized)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    scheduler.set_timesteps(denoising_steps)
    
    # Initialize pure noise for u with correct scale
    B, _, H, W = cond.shape
    samples = torch.randn((B, 1, H, W), device=device)
    if hasattr(scheduler, 'init_noise_sigma'):
        samples = samples * scheduler.init_noise_sigma
    
    print(f"Generating {B} samples with {denoising_steps} denoising steps...")
    
    with torch.no_grad():
        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising")):
            # Standard denoising step
            model_input = samples
            if hasattr(scheduler, 'scale_model_input'):
                model_input = scheduler.scale_model_input(model_input, t)
            x_in = torch.cat([model_input, cond], dim=1)  # [B,5,H,W]
            # Predict noise on u
            noise_pred = model(x_in, t).sample        # [B,1,H,W]
            # Take one denoising step
            samples = scheduler.step(noise_pred, t, samples).prev_sample
            # Trust the model's physics learning - no intermediate BC enforcement
    
    # Final denormalization
    samples = samples * (sigma + 1e-6) + mu
    
    # Extract boundary condition information from conditioning
    bmask = cond[:, 1:2]  # boundary mask  
    d_norm = cond[:, 2:3]  # normalized Dirichlet BC
    # Denormalize boundary conditions
    dirichlet_bc = d_norm * (sigma + 1e-6) + mu
    # Final boundary condition enforcement
    samples = torch.where(bmask == 1, dirichlet_bc, samples)
    
    return samples