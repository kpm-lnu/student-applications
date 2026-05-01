import torch
from torch.utils.data import Dataset

class HarmonicFieldDataset(Dataset):
    def __init__(self, data_list, mu_u, sigma_u):
        """
        HarmonicFieldDataset for loading harmonic function fields and boundary conditions.
        - data_list: list of dicts with keys 'u', 'dirichlet', 'neumann'
        - mu_u: global mean of u values for normalization
        - sigma_u: global std of u values for normalization
        - gmask: geometry mask tensor
        - bmask: boundary mask tensor
        """
        self.data = data_list
        self.mu = mu_u
        self.sig = sigma_u

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 1) Normalize u
        u = torch.tensor(sample['u'], dtype=torch.float32)
        u = (u - self.mu) / (self.sig + 1e-6)          # [H,W]
        
        # 2) Normalize BC values
        d_norm = torch.tensor((sample['dirichlet'] - self.mu) / (self.sig + 1e-6), dtype=torch.float32)
        n_norm = torch.tensor(sample['neumann'] / (self.sig + 1e-6), dtype=torch.float32)
        
        # Stack cond maps into [4,H,W]
        gmask = torch.tensor(sample['gmask'], dtype=torch.float32)
        bmask = torch.tensor(sample['bmask'], dtype=torch.float32)
        cond = torch.stack([gmask, bmask, d_norm, n_norm], dim=0)  # [4,H,W]

        # Return u channel and cond
        return u.unsqueeze(0), cond
