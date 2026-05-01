import matplotlib.pyplot as plt
import numpy as np

from harmonic.sample_harmonic_field import sample_harmonic, compute_normal_derivative
from grid.compute_grid import *

# Parameterize Г1, Г2
M     = 128
t     = np.linspace(0, 2*np.pi, M, endpoint=True)

x_Γ1  = 1.3*np.cos(t)
y_Γ1  = np.sin(t)

x_Γ2  = 0.5*np.cos(t)
y_Γ2  = 0.4*np.sin(t) - 0.3*np.sin(t)**2

N         = 8
pixel_res = 128

square_bounds = (-1.5, 1.5, -1.5, 1.5)

# Compute the geometry mask
geometry_mask, x_grid, y_grid = compute_geometry_mask(
  outer_curve_x=x_Γ1,
  outer_curve_y=y_Γ1,
  inner_curves_x=x_Γ2,
  inner_curves_y=y_Γ2,
  square_bounds=square_bounds,
  pixel_res=pixel_res,
  exclude_boundary=True)

fig1, ax1 = visualize_geometry_mask(
  geometry_mask,
  x_grid,
  y_grid,
  outer_curve_x=x_Γ1,
  outer_curve_y=y_Γ1,
  inner_curve_x=x_Γ2,
  inner_curve_y=y_Γ2,
  title="Geometry Mask"
)

# Compute the boundary mask
boundary_mask, x_grid, y_grid = compute_boundary_mask(
  outer_curve_x=x_Γ1,
  outer_curve_y=y_Γ1,
  square_bounds=square_bounds,
  pixel_res=pixel_res,
)

# Compute the boundary mask with values for u(x, y) and du/dn(x, y)

boundary_indices = np.where(boundary_mask == 1)
boundary_x       = x_grid[boundary_indices]
boundary_y       = y_grid[boundary_indices]

print(f"Number of boundary points: {len(boundary_x)}")

a              = np.random.randn(N) # TODO: sample many harmonic functions
u_boundary     = sample_harmonic(boundary_x, boundary_y, a)
du_dn_boundary = compute_normal_derivative(boundary_x, boundary_y, a) # Uses analytical tangents for Γ1

# Filter out invalid values
valid_indices = np.isfinite(u_boundary) & np.isfinite(du_dn_boundary)

print(f"Valid boundary points after filtering: {np.sum(valid_indices)}")

boundary_dirichlet_mask = np.zeros_like(boundary_mask)
boundary_dirichlet_mask[boundary_indices] = u_boundary

boundary_neumann_mask = np.zeros_like(boundary_mask)
boundary_neumann_mask[boundary_indices] = du_dn_boundary

# Print statistics
print(f"Dirichlet values - min: {np.nanmin(u_boundary):.3f}, max: {np.nanmax(u_boundary):.3f}, mean: {np.nanmean(u_boundary):.3f}")
print(f"Neumann values - min: {np.nanmin(du_dn_boundary):.3f}, max: {np.nanmax(du_dn_boundary):.3f}, mean: {np.nanmean(du_dn_boundary):.3f}")

# Check for zero values
zero_dirichlet = np.sum(u_boundary == 0)
zero_neumann = np.sum(du_dn_boundary == 0)
print(f"Zero Dirichlet values: {zero_dirichlet}")
print(f"Zero Neumann values: {zero_neumann}")

# Check for values that would be excluded by != 0 condition
nonzero_dirichlet = np.sum(boundary_dirichlet_mask != 0)
nonzero_neumann = np.sum(boundary_neumann_mask != 0)
print(f"Non-zero values in Dirichlet mask: {nonzero_dirichlet}")
print(f"Non-zero values in Neumann mask: {nonzero_neumann}")

# Compute u_true on the interior grid
interior_indices = np.where(geometry_mask == 1)
interior_x = x_grid[interior_indices]
interior_y = y_grid[interior_indices]

print(f"Number of interior points: {len(interior_x)}")

# Sample the harmonic field at interior points using the same coefficients 'a'
u_true_interior = sample_harmonic(interior_x, interior_y, a)

# Create full u_true mask (interior values only)
u_true_mask = np.zeros_like(geometry_mask, dtype=float)
u_true_mask[interior_indices] = u_true_interior

fig2, ax2 = visualize_boundary_mask(
  boundary_mask,
  x_grid,
  y_grid,
  title="Boundary Mask"
)

# Visualize Dirichlet boundary conditions
fig3, ax3 = visualize_dirichlet_mask(
  boundary_dirichlet_mask,
  x_grid,
  y_grid,
  boundary_mask=boundary_mask,
  title="Dirichlet Boundary Conditions (u values)"
)

# Visualize Neumann boundary conditions
fig4, ax4 = visualize_neumann_mask(
  boundary_neumann_mask,
  x_grid,
  y_grid,
  boundary_mask=boundary_mask,
  title="Neumann Boundary Conditions (∂u/∂n values)"
)

# Visualize both boundary conditions side by side
fig5, (ax5a, ax5b) = visualize_boundary_conditions_combined(
  boundary_dirichlet_mask,
  boundary_neumann_mask,
  x_grid,
  y_grid,
  boundary_mask=boundary_mask,
  title="Boundary Conditions Comparison"
)


# Visualize the true solution field
fig6, ax6 = visualize_interior_solution(
    u_true_mask,
    x_grid, 
    y_grid,
    geometry_mask=geometry_mask,
    title="True Solution u(x,y) on Interior"
)

plt.show()