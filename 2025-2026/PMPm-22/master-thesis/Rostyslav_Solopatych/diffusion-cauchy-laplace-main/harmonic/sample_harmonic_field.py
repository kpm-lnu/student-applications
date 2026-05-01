import numpy as np
from torch import sqrt, Tensor

def compute_harmonic_basis(x, y, N):
    """
    Return an array shape (N, len(x)) where row n is u_{n+1}(x,y).
    """
    z = x + 1j * y
    basis = []
    for n in range(1, N+1):
        # complex power
        z_pow = z**n
        basis.append(z_pow.real)    # u_n = Re(z^n)
    return np.stack(basis, axis=0)  # shape (N, len(xs))


def sample_harmonic(x, y, a):
    """
    Generate one harmonic snapshot on points (x,y).
    
    Parameters:
    - x, y: coordinate arrays for evaluation points
    - a: coefficients array of shape (N,)
    
    Returns:
    - u: harmonic field values at (x,y)
    """
    N = len(a)
    basis = compute_harmonic_basis(x, y, N) # (N, P)
    u = a @ basis                           # linear combination
    return u                                # shape (P,)


def compute_harmonic_gradients(x, y, a):
    """
    Compute the gradients of the harmonic field with respect to x and y.
    Returns two arrays (du_dx, du_dy) each of shape (P,).
    """
    N = a.shape[0]

    z = x + 1j * y
    dz_dx = 1
    dz_dy = 1j

    basis_dx = []
    basis_dy = []
    for n in range(1, N+1):
        z_pow = z**n
        dz_pow_dx = n * z**(n-1) * dz_dx
        dz_pow_dy = n * z**(n-1) * dz_dy

        basis_dx.append(dz_pow_dx.real)     # du_n/dx = Re(dz^n/dx)
        basis_dy.append(dz_pow_dy.real)     # du_n/dy = Re(dz^n/dy)

    basis_dx = np.stack(basis_dx, axis=0)   # shape (N, len(x))
    basis_dy = np.stack(basis_dy, axis=0)   # shape (N, len(x))

    du_dx = a @ basis_dx        # linear combination for x-gradient
    du_dy = a @ basis_dy        # linear combination for y-gradient

    return du_dx, du_dy


def compute_normal_derivative(x, y, a):
    """
    Compute the normal derivative of the harmonic field at Γ1 boundary points.
    
    Parameters:
    - x, y: arrays of boundary points (assumed to be from Γ1: x=1.3*cos(t), y=sin(t))
    - a: coefficients of the harmonic expansion
    
    Returns:
    - du_dn: normal derivative at each point (positive pointing outward)
    """
    du_dx, du_dy = compute_harmonic_gradients(x, y, a)
    n_x, n_y = compute_normals(x, y)

    # Normal derivative ∂u/∂n = ∇u · n̂
    du_dn = du_dx * n_x + du_dy * n_y
    return du_dn


def compute_normals(x, y, device=None):
    """
    Compute outward normal vectors at boundary points (x,y) on Γ1.
    
    Parameters:
    - x, y: arrays of boundary points (assumed to be from Γ1: x=1.3*cos(t), y=sin(t))
    - device: optional torch.device to place tensors on when torch_tensor=True

    Returns:
    - n_x, n_y: arrays of normal vector components at each boundary point
    """
    # For ellipse F(x,y) = (x/1.3)² + y² - 1 = 0
    # Outward normal = ∇F = (2x/1.3², 2y)
    if device is not None:
        import torch
        # Ensure tensors on the requested device, and keep type consistent with inputs if possible
        x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32, device=device)
        y_t = y if isinstance(y, torch.Tensor) else torch.as_tensor(y, dtype=torch.float32, device=device)

        n_x = 2.0 * x_t / (1.3**2)
        n_y = 2.0 * y_t
        norm = torch.sqrt(n_x**2 + n_y**2).clamp_min(1e-10)
        n_x = n_x / norm
        n_y = n_y / norm
        return n_x, n_y
    else:
        n_x = 2 * x / (1.3**2)
        n_y = 2 * y
        norm = np.sqrt(n_x**2 + n_y**2) + 1e-10  # avoid division by zero
        n_x = n_x / norm
        n_y = n_y / norm
        return n_x, n_y