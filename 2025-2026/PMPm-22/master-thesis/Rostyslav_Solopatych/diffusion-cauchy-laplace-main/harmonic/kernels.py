import torch

def get_laplacian_kernel(h: float, device=None):
    """
    Returns a 3×3 discrete Laplacian kernel scaled by 1/h², shaped [1,1,3,3]
    for direct use in F.conv2d. Approximates Δu via the standard 5-point stencil:

        Δu_{i,j} ≈ (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4·u_{i,j}) / h²

    Parameters:
    - h: scalar grid spacing (assumed equal in x and y).
    - device: torch device for the tensor.
    """
    k = torch.tensor(
        [[0.,  1., 0.],
         [1., -4., 1.],
         [0.,  1., 0.]],
        dtype=torch.float32,
        device=device
    ) / (h * h)
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
