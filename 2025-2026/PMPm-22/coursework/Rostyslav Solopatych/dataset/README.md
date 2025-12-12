# Harmonic Dataset Generation

This module generates large-scale datasets of harmonic functions with boundary conditions for training the diffusion model.

## Overview

The dataset generation creates:

- **Interior solutions** `u(x,y)` on the geometry domain
- **Dirichlet boundary conditions** `u(x,y)` on the boundary Γ1
- **Neumann boundary conditions** `∂u/∂n(x,y)` on the boundary Γ1

Each sample is a harmonic function generated using a random linear combination of harmonic basis functions. **The key innovation is using diverse coefficient sampling strategies** to ensure the dataset covers a wide range of boundary condition types, from smooth to highly oscillatory, and from localized to global patterns. This maximizes the diffusion model's ability to generalize to various boundary conditions.
Each sample includes the geometry and boundary masks directly, making integration with the training pipeline more streamlined and eliminating the need to separately compute or manage these masks.

## Files

- `generate_harmonic_dataset.py` - Main dataset generation script
- `test_dataset_generation.py` - Test suite for validation
- `harmonic_field_dataset.py` - Class inheriting torch Dataset
- `compute_stats.py` - utilities for computing mean and standard deviation for dataset normalization

## Quick Start

### Generate a test dataset (small)

```bash
python test_dataset_generation.py
```

### Generate full dataset (100k samples)

```bash
python generate_harmonic_dataset.py --num_samples 100000 --pixel_res 64 --output harmonic_field_dataset.pt
```

### Generate large dataset (200k samples)

```bash
python generate_harmonic_dataset.py --num_samples 200000 --pixel_res 64 --N 32
```

## Command Line Options

```bash
python generate_harmonic_dataset.py [OPTIONS]
```

**Options:**

- `--num_samples`: Number of samples to generate (default: 1000)
- `--pixel_res`: Grid resolution NxN (default: 64)
- `--N`: Number of harmonic basis functions (default: 8)
- `--output`: Output filename (default: harmonic_field_dataset.pt)
- `--seed`: Random seed for reproducibility (default: 42)
- `--val_split`: Fraction of data for validation set (default: 0.2)

## Data Format

The generated dataset is saved as a list of dictionaries, where each dictionary represents one sample:

```python
sample = {
    'u': np.ndarray,           # Shape (H, W) - Full solution on grid
    'gmask': np.ndarray,       # Shape (H, W) - Geometry mask (1=interior, 0=exterior)
    'bmask': np.ndarray,       # Shape (H, W) - Boundary mask (1=boundary, 0=elsewhere)
    'dirichlet': np.ndarray,   # Shape (H, W) - Dirichlet BC values
    'neumann': np.ndarray      # Shape (H, W) - Neumann BC values
}
```

**Data Details:**

- `u`: Interior values of the harmonic function
- `gmask`: Geometry mask indicating the valid domain (1=interior, 0=exterior)
- `bmask`: Boundary mask indicating boundary points (1=boundary, 0=elsewhere)
- `dirichlet`: Dirichlet boundary condition values (zero outside boundary)
- `neumann`: Neumann boundary condition values (zero outside boundary)
- All arrays are `np.float32` and have the same shape `(pixel_res, pixel_res)`

## Geometry Setup

- **Outer boundary Γ1**: `x = 1.3*cos(t)`, `y = sin(t)`
- **Inner boundary Γ2**: `x = 0.5*cos(t)`, `y = 0.4*sin(t) - 0.3*sin(t)²`
- **Domain**: Square `[-1.5, 1.5] × [-1.5, 1.5]`
- **Grid**: `pixel_res × pixel_res` uniform grid

## Harmonic Function Generation

Each harmonic function is generated as linear combination of harmonic base functions:

```
u(x,y) = Σ(n=1 to N) a_n * Re(z^n)
```

Where:

- `z = x + iy` (complex coordinate)
- `a_n` are random coefficients using **diverse sampling strategies** for maximum boundary condition variety:
  - **Sparse**: Few dominant terms (focused solutions)
  - **Concentrated**: Power-law distributed (realistic frequency decay)
  - **Scaled**: Variable amplitude scaling (small to very large solutions)
  - **Oscillatory**: Alternating signs with decay (wave-like patterns)
  - **Clustered**: Concentrated in frequency bands
  - **Mixed**: Random selection of above strategies for maximum diversity
- `N` is the number of basis functions (default: 8)

**Boundary Conditions:**

- **Dirichlet**: `u(x,y)` evaluated on boundary points
- **Neumann**: `∂u/∂n(x,y)` computed analytically using the known geometry

### Diversity Strategies

To ensure robust diffusion model training, the coefficient sampling uses multiple strategies:

1. **Sparse Solutions** (1-8 active terms): Creates focused, localized boundary conditions
2. **Power-Law Decay** (α ∈ [0.5, 2.5]): Realistic frequency content with higher modes decaying
3. **Variable Scaling** (0.1× to 10×): Covers small to very large amplitude solutions
4. **Oscillatory Patterns**: Alternating signs create wave-like boundary conditions
5. **Frequency Clustering**: Concentrates energy in specific frequency bands
6. **Mixed Mode**: Randomly selects from all strategies for maximum diversity

This approach ensures the dataset covers:

- ✅ **Smooth vs. highly oscillatory** boundary conditions
- ✅ **Small vs. large amplitude** solutions
- ✅ **Localized vs. global** solution patterns
- ✅ **Simple vs. complex** frequency content

## Memory Usage

Dataset with 100k training samples for 64x64 grid ~ 8GB

## Output Files

The script generates three files with train/validation split:

1. **Training dataset**: `<output>_train.pt` (e.g., `harmonic_field_dataset_train.pt`)

   - Training samples (default: 80% of data)
   - List of sample dictionaries
   - Compatible with `HarmonicFieldDataset` class

2. **Validation dataset**: `<output>_val.pt` (e.g., `harmonic_field_dataset_val.pt`)

   - Validation samples (default: 20% of data)
   - Same format as training set
   - Used for monitoring generalization

3. **Metadata**: `<output>_metadata.pt` (e.g., `harmonic_field_dataset_metadata.pt`)
   - Dataset statistics (computed from training set only)
   - Generation parameters
   - Validation results
   - Train/validation split sizes

**Important**: Normalization statistics (μ, σ) are computed from the training set only to prevent data leakage.

## Validation

Each sample undergoes validation:

- ✅ Finite values check
- ✅ Boundary coverage > 80%
- ✅ Interior coverage > 80%
- ✅ Reasonable value ranges

Invalid samples are automatically replaced.

## Integration with Training

The generated dataset is directly compatible with the training pipeline:

```python
# Load train and validation datasets
train_data = torch.load("harmonic_field_dataset_train.pt", weights_only=False)
val_data = torch.load("harmonic_field_dataset_val.pt", weights_only=False)

# Load normalization statistics (computed from training set only)
metadata = torch.load('harmonic_field_dataset_metadata.pt')
mu = metadata['global_mu']
sig = metadata['global_sigma']

# Extract geometry and boundary masks from first sample (they're identical for all samples)
sample = train_data[0]
gmask = torch.tensor(sample['gmask'], dtype=torch.float32)
bmask = torch.tensor(sample['bmask'], dtype=torch.float32)

# Create datasets for training and validation
train_dataset = HarmonicFieldDataset(train_data, mu, sig, gmask, bmask)
val_dataset = HarmonicFieldDataset(val_data, mu, sig, gmask, bmask)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
```

See `TRAIN_VAL_SPLIT_GUIDE.md` in the repository root for detailed information about the train/validation split implementation.

## Performance Tips

**For faster generation:**

- Use smaller `--N` (fewer basis functions)
- Use smaller `--pixel_res` for testing

**For higher quality:**

- Use larger `--N` (more complex harmonics)
- Use higher `--pixel_res` resolution
- Generate more samples

## Example Usage

```bash
# Generate test dataset (80/20 train/val split)
python generate_harmonic_dataset.py --num_samples 1000 --pixel_res 32 --output test_small.pt
# Creates: test_small_train.pt (800 samples), test_small_val.pt (200 samples)

# Generate production dataset with custom validation split
python generate_harmonic_dataset.py --num_samples 150000 --pixel_res 64 --N 8 --val_split 0.15
# Creates: harmonic_field_dataset_train.pt (127,500 samples), harmonic_field_dataset_val.pt (22,500 samples)

# Generate high-resolution dataset
python generate_harmonic_dataset.py --num_samples 50000 --pixel_res 128 --N 8
# Creates: harmonic_field_dataset_train.pt (40,000 samples), harmonic_field_dataset_val.pt (10,000 samples)
```

## Troubleshooting

**Common Issues:**

1. **Memory Error**: Reduce `--num_samples` or use a system with more RAM
2. **Invalid Samples**: Check geometry setup and boundary tolerance
3. **Slow Generation**: Reduce `--N` or `--pixel_res` for testing

**Verification:**

```bash
# Run test suite
python test_dataset_generation.py

# Check dataset compatibility
python -c "
import torch
from model.harmonic_field_dataset import HarmonicFieldDataset
data = torch.load('harmonic_field_dataset.pt')
print(f'Loaded {len(data)} samples')
print(f'Sample keys: {list(data[0].keys())}')
print(f'Sample shapes:')
for key in data[0].keys():
    print(f'  {key}: {data[0][key].shape}')
"
```

---

## Technical Details

**Coordinate System:**

- Grid uses `indexing='xy'` convention
- Boundary normal vectors point outward

**Numerical Stability:**

- Boundary detection with adaptive tolerance
- Gradient magnitude clamping for normal computation
- Invalid value replacement with zeros

**File Format:**

- PyTorch `.pt` format using `torch.save()`
- Cross-platform compatible
- Efficient loading with `torch.load()`
