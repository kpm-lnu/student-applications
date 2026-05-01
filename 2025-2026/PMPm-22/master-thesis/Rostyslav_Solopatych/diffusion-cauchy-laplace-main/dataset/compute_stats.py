import numpy as np
from tqdm import tqdm

def compute_global_stats(dataset):
    """
    Compute normalization statistics from all values in the dataset.
    
    Parameters:
    - dataset: loaded dataset (list of dicts)
    
    Returns:
    - (mu, sigma): mean and standard deviation computed from all values
    """
    import numpy as np
    
    print(f"Computing normalization statistics from all values...")
    
    # Compute statistics from all values
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for sample in tqdm(dataset, desc="Computing μ,σ"):
        u = sample['u'].astype(np.float64)
        
        total_sum += u.sum()
        total_sq_sum += (u * u).sum()
        total_count += u.size

    mu = total_sum / total_count
    var = total_sq_sum / total_count - mu**2
    sigma = np.sqrt(var)
    
    print(f"Statistics: μ={mu:.6f}, σ={sigma:.6f}")
    
    return float(mu), float(sigma)