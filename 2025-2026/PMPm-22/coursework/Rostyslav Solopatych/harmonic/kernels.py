import torch

def get_laplacian_kernel(device=None):
    """
    Returns a 3×3 Laplacian kernel for Δ without 1/h²,
    shaped [1,1,3,3] for direct use in F.conv2d.
    
    Note: For physics-informed training, we use normalized scaling
    to avoid numerical conditioning issues with small h.
    """
    k = torch.tensor(
        [[0.,  1., 0.],
         [1., -4., 1.],
         [0.,  1., 0.]],
        dtype=torch.float32,
        device=device
    )
    # Use normalized scaling instead of 1/h² for better numerical stability
    # The physics loss will be relative, which is more appropriate for training
    return k.view(1, 1, 3, 3)

def get_neumann_kernel(h: float, device=None):
    """
    Returns central‐difference kernels for ∂u/∂x and ∂u/∂y,
    each shaped [1,1,3,3], scaled by 1/(2h) where h is grid spacing.

    Parameters:
    - h: scalar spacing to use for both x and y (convenience).
    - device: torch device for tensors.
    """
    normalization_factor = 1.0 / (2.0 * h)
    
    # ∂/∂x central difference
    dx = torch.tensor(
        [[0., 0., 0.],
         [-1., 0., 1.],
         [0., 0., 0.]],
        dtype=torch.float32,
        device=device
    ) * normalization_factor

    # ∂/∂y central difference
    dy = torch.tensor(
        [[0., -1., 0.],
         [0.,  0., 0.],
         [0.,  1., 0.]],
        dtype=torch.float32,
        device=device
    ) * normalization_factor

    return dx.view(1,1,3,3), dy.view(1,1,3,3)
