#!/usr/bin/env python3
"""Compare DDPM-based and integral-method reconstructions of u on the same test dataset sample.

For each sample index, both methods receive the *same* boundary Cauchy data
(``dirichlet``/``neumann`` pixel maps, optionally noised). We report MSE/RMSE/L2/L∞ for
each method against ground-truth ``u``, plus the head-to-head DDPM-vs-integral discrepancy.

Usage
-----
    python verify_ddpm_vs_integral.py \\
        --dataset ./harmonic_field_dataset_val.pt \\
        --checkpoint ./checkpoints/cond_ddpm_best \\
        --sample_idx 0 1 2 \\
        --ensemble_size 10 \\
        --M 32
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.generate_samples import generate_samples, load_trained_model
from integral_method.from_dataset import reconstruct_from_dataset_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def auto_detect_scheduler_config(checkpoint_path: str) -> str:
    name = os.path.basename(checkpoint_path.rstrip("/"))
    parent = os.path.dirname(checkpoint_path)
    if name == "cond_ddpm_best":
        return os.path.join(parent, "scheduler_best", "scheduler_config.json")
    try:
        epoch_num = int(name.split("epoch")[-1])
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Cannot auto-detect scheduler config from {checkpoint_path}; pass --scheduler_config."
        ) from e
    return os.path.join(parent, f"scheduler_epoch{epoch_num}.json", "scheduler_config.json")


def metadata_path_for(dataset_file: str) -> str:
    if "_train.pt" in dataset_file:
        return dataset_file.replace("_train.pt", "_metadata.pt")
    if "_val.pt" in dataset_file:
        return dataset_file.replace("_val.pt", "_metadata.pt")
    return dataset_file.replace(".pt", "_metadata.pt")


def maybe_add_noise(
    dirichlet: np.ndarray, neumann: np.ndarray, noise: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if noise <= 0:
        return dirichlet, neumann
    d_range = float(np.max(dirichlet) - np.min(dirichlet))
    n_range = float(np.max(neumann) - np.min(neumann))
    d = dirichlet + rng.normal(0.0, noise * d_range, dirichlet.shape)
    n = neumann + rng.normal(0.0, noise * n_range, neumann.shape)
    return d, n


def compute_metrics(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> dict:
    p = pred[mask]
    r = ref[mask]
    diff = p - r
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    l2 = float(np.linalg.norm(diff))
    l2_rel = float(l2 / (np.linalg.norm(r) + 1e-8))
    linf = float(np.max(np.abs(diff)))
    linf_rel = float(linf / (np.max(np.abs(r)) + 1e-8))
    return {
        "mse": mse,
        "rmse": rmse,
        "l2": l2,
        "l2_rel": l2_rel,
        "linf": linf,
        "linf_rel": linf_rel,
    }


def run_ddpm(
    sample: dict,
    metadata: dict,
    model,
    scheduler,
    device: torch.device,
    dirichlet: np.ndarray,
    neumann: np.ndarray,
    ensemble_size: int,
    denoising_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ensemble_mean, ensemble_std) as 2D numpy arrays."""
    mu = metadata["global_mu"]
    sigma = metadata["global_sigma"]
    d_norm = torch.tensor((dirichlet - mu) / (sigma + 1e-6), dtype=torch.float32)
    n_norm = torch.tensor(neumann / (sigma + 1e-6), dtype=torch.float32)
    gmask = torch.tensor(sample["gmask"], dtype=torch.float32)
    bmask = torch.tensor(sample["bmask"], dtype=torch.float32)
    cond_single = torch.stack([gmask, bmask, d_norm, n_norm], dim=0).unsqueeze(0).to(device)
    cond = cond_single.repeat(ensemble_size, 1, 1, 1)
    samples = generate_samples(
        model=model,
        scheduler=scheduler,
        cond=cond,
        mu=mu,
        sigma=sigma,
        denoising_steps=denoising_steps,
        device=device,
    )
    mean = torch.mean(samples, dim=0, keepdim=True)[0, 0].cpu().numpy()
    std = torch.std(samples, dim=0, keepdim=True)[0, 0].cpu().numpy()
    return mean, std


def _fmt_metrics(m: dict) -> str:
    """One-line summary of a metrics dict for embedding in a subplot title."""
    return (
        f"RMSE={m['rmse']:.3e}  L2={m['l2']:.3e}\n"
        f"L2rel={m['l2_rel']*100:.2f}%  L∞={m['linf']:.3e}  L∞rel={m['linf_rel']*100:.2f}%"
    )


def make_comparison_figure(
    u_gt: np.ndarray,
    u_ddpm: np.ndarray,
    u_int: np.ndarray,
    mask: np.ndarray,
    metrics_ddpm: dict,
    metrics_int: dict,
    metrics_dvi: dict,
    title_suffix: str,
    output_path: str,
) -> None:
    def masked(arr):
        out = np.where(mask, arr, np.nan)
        return out

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    sol_min = float(np.nanmin([masked(u_gt), masked(u_ddpm), masked(u_int)]))
    sol_max = float(np.nanmax([masked(u_gt), masked(u_ddpm), masked(u_int)]))
    diff_ddpm = u_ddpm - u_gt
    diff_int = u_int - u_gt
    diff_dvi = u_ddpm - u_int
    diff_max = max(
        float(np.nanmax(np.abs(masked(diff_ddpm)))),
        float(np.nanmax(np.abs(masked(diff_int)))),
        float(np.nanmax(np.abs(masked(diff_dvi)))),
        1e-8,
    )

    range_str = f"range=[{sol_min:.3f}, {sol_max:.3f}]"
    panels = [
        (axes[0, 0], masked(u_gt), f"Ground truth\n{range_str}", "RdBu_r", sol_min, sol_max),
        (axes[0, 1], masked(u_ddpm), f"DDPM ensemble mean\n{_fmt_metrics(metrics_ddpm)}", "RdBu_r", sol_min, sol_max),
        (axes[0, 2], masked(u_int), f"Integral method\n{_fmt_metrics(metrics_int)}", "RdBu_r", sol_min, sol_max),
        (axes[1, 0], masked(diff_ddpm), f"DDPM − GT\n{_fmt_metrics(metrics_ddpm)}", "RdBu_r", -diff_max, diff_max),
        (axes[1, 1], masked(diff_int), f"Integral − GT\n{_fmt_metrics(metrics_int)}", "RdBu_r", -diff_max, diff_max),
        (axes[1, 2], masked(diff_dvi), f"DDPM − Integral\n{_fmt_metrics(metrics_dvi)}", "RdBu_r", -diff_max, diff_max),
    ]
    for ax, data, title, cmap, vmin, vmax in panels:
        im = ax.imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Footer with head-to-head winner — easy to read at a glance.
    if metrics_ddpm["rmse"] < metrics_int["rmse"]:
        winner = f"DDPM wins on RMSE ({metrics_ddpm['rmse']:.3e} < {metrics_int['rmse']:.3e})"
    elif metrics_int["rmse"] < metrics_ddpm["rmse"]:
        winner = f"Integral wins on RMSE ({metrics_int['rmse']:.3e} < {metrics_ddpm['rmse']:.3e})"
    else:
        winner = "tie on RMSE"
    fig.suptitle(f"{title_suffix}   |   {winner}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def append_csv(rows: list[dict], csv_path: str) -> None:
    fieldnames = [
        "sample_idx",
        "noise_level",
        "method",
        "mse",
        "rmse",
        "l2",
        "l2_rel",
        "linf",
        "linf_rel",
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a" if file_exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Per-sample driver
# ---------------------------------------------------------------------------
def process_sample(
    sample_idx: int,
    dataset,
    metadata: dict,
    model,
    scheduler,
    device: torch.device,
    args,
    rng: np.random.Generator,
) -> dict:
    sample = dataset[sample_idx]
    u_gt = np.asarray(sample["u"], dtype=np.float64)
    gmask = np.asarray(sample["gmask"])
    bmask = np.asarray(sample["bmask"])
    dirichlet_clean = np.asarray(sample["dirichlet"], dtype=np.float64)
    neumann_clean = np.asarray(sample["neumann"], dtype=np.float64)

    dirichlet, neumann = maybe_add_noise(dirichlet_clean, neumann_clean, args.noise, rng)

    # --- DDPM branch -------------------------------------------------------
    logger.info(f"[sample {sample_idx}] running DDPM ensemble (size={args.ensemble_size})...")
    u_ddpm, ddpm_std = run_ddpm(
        sample=sample,
        metadata=metadata,
        model=model,
        scheduler=scheduler,
        device=device,
        dirichlet=dirichlet,
        neumann=neumann,
        ensemble_size=args.ensemble_size,
        denoising_steps=args.num_diffusion_timesteps,
    )

    # --- Integral-method branch -------------------------------------------
    logger.info(f"[sample {sample_idx}] running integral method (M={args.M})...")
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    integral = reconstruct_from_dataset_sample(
        dirichlet=dirichlet,
        neumann=neumann,
        gmask=gmask,
        bmask=bmask,
        M=args.M,
        lam=args.lam,
    )
    u_int = integral.u_image

    # --- Metrics ----------------------------------------------------------
    eval_mask = ((gmask.astype(bool)) | (bmask.astype(bool)))
    m_ddpm = compute_metrics(u_ddpm, u_gt, eval_mask)
    m_int = compute_metrics(u_int, u_gt, eval_mask)
    m_dvi = compute_metrics(u_ddpm, u_int, eval_mask)

    logger.info(
        f"[sample {sample_idx}] DDPM   "
        f"RMSE={m_ddpm['rmse']:.4e}  L2rel={m_ddpm['l2_rel']*100:.2f}%  "
        f"Linf={m_ddpm['linf']:.4e}  LinfRel={m_ddpm['linf_rel']*100:.2f}%"
    )
    logger.info(
        f"[sample {sample_idx}] INTEG  "
        f"RMSE={m_int['rmse']:.4e}  L2rel={m_int['l2_rel']*100:.2f}%  "
        f"Linf={m_int['linf']:.4e}  LinfRel={m_int['linf_rel']*100:.2f}%   "
        f"(λ={integral.lam:.2e}, cond(A)={integral.cond_A:.2e})"
    )
    logger.info(
        f"[sample {sample_idx}] DDPM−INTEG  "
        f"RMSE={m_dvi['rmse']:.4e}  Linf={m_dvi['linf']:.4e}"
    )

    # --- Persist ----------------------------------------------------------
    noise_tag = f"_noise{args.noise}" if args.noise > 0 else ""
    base = os.path.join(args.output_dir, f"sample_{sample_idx}_ddpm_vs_integral{noise_tag}")

    make_comparison_figure(
        u_gt=u_gt,
        u_ddpm=u_ddpm,
        u_int=u_int,
        mask=eval_mask,
        metrics_ddpm=m_ddpm,
        metrics_int=m_int,
        metrics_dvi=m_dvi,
        title_suffix=(
            f"Sample {sample_idx}"
            + (f"  |  noise={args.noise*100:.1f}%" if args.noise > 0 else "")
        ),
        output_path=base + ".png",
    )

    np.savez(
        base + ".npz",
        u_gt=u_gt,
        u_ddpm=u_ddpm,
        u_int=u_int,
        ddpm_std=ddpm_std,
        gmask=gmask,
        bmask=bmask,
        dirichlet=dirichlet,
        neumann=neumann,
        f1=integral.f1,
        f2=integral.f2,
        psi0=integral.psi0,
        psi1=integral.psi1,
        lam=integral.lam,
        cond_A=integral.cond_A,
    )

    json_payload = {
        "sample_idx": int(sample_idx),
        "noise_level": float(args.noise),
        "M": int(args.M),
        "ensemble_size": int(args.ensemble_size),
        "lam": float(integral.lam),
        "cond_A": float(integral.cond_A),
        "metrics": {
            "ddpm_vs_gt": m_ddpm,
            "integral_vs_gt": m_int,
            "ddpm_vs_integral": m_dvi,
        },
    }
    with open(base + ".json", "w") as f:
        json.dump(json_payload, f, indent=2)

    csv_rows = []
    for method_name, m in [("ddpm", m_ddpm), ("integral", m_int), ("ddpm_vs_integral", m_dvi)]:
        csv_rows.append({
            "sample_idx": sample_idx,
            "noise_level": args.noise,
            "method": method_name,
            **{k: f"{v:.6e}" for k, v in m.items()},
        })
    append_csv(csv_rows, os.path.join(args.output_dir, "ddpm_vs_integral_results.csv"))

    return json_payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=str, default="./verification_results")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/cond_ddpm_best")
    parser.add_argument("--scheduler_config", type=str, default=None)
    parser.add_argument("--sample_idx", type=int, nargs="+", default=[0])
    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--num_diffusion_timesteps", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--M", type=int, default=32, help="half number of integral-method nodes")
    parser.add_argument("--lam", type=float, default=None, help="override Tikhonov λ (else L-curve)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    scheduler_config_path = args.scheduler_config or auto_detect_scheduler_config(args.checkpoint)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    if not os.path.exists(scheduler_config_path):
        raise FileNotFoundError(scheduler_config_path)
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(args.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, scheduler = load_trained_model(
        checkpoint_path=args.checkpoint,
        scheduler_config_path=scheduler_config_path,
        device=device,
    )

    dataset = torch.load(args.dataset, weights_only=False)
    metadata = torch.load(metadata_path_for(args.dataset), weights_only=False)
    logger.info(
        f"Loaded {len(dataset)} samples; μ={metadata['global_mu']:.6f}, σ={metadata['global_sigma']:.6f}"
    )

    rng = np.random.default_rng(args.seed)

    summaries = []
    for sample_idx in args.sample_idx:
        if sample_idx < 0 or sample_idx >= len(dataset):
            logger.error(f"Sample index {sample_idx} out of range; skipping.")
            continue
        summaries.append(process_sample(
            sample_idx=sample_idx,
            dataset=dataset,
            metadata=metadata,
            model=model,
            scheduler=scheduler,
            device=device,
            args=args,
            rng=rng,
        ))

    logger.info("=" * 80)
    logger.info(f"Processed {len(summaries)} sample(s). Results in {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
