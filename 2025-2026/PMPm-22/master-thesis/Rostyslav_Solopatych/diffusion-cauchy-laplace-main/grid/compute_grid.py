import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from torch import linspace, meshgrid


# region Geometry mask

def point_in_polygon(x, y, poly_x, poly_y, exclude_boundary=True, boundary_tolerance=1e-10):
    """
    Determine if points (x,y) are inside a polygon defined by vertices (poly_x, poly_y).
    Uses the ray casting algorithm via matplotlib.path.Path.
    
    Parameters:
    - x, y: arrays of query points
    - poly_x, poly_y: arrays defining the polygon vertices
    - exclude_boundary: if True, points on the boundary are excluded (returned as False)
    - boundary_tolerance: tolerance for boundary detection
    
    Returns:
    - mask: boolean array, True if point is inside polygon (and not on boundary if exclude_boundary=True)
    """
    # Create polygon path
    vertices = np.column_stack((poly_x, poly_y))
    polygon_path = Path(vertices)
    
    # Query points
    points = np.column_stack((x.flatten(), y.flatten()))
    
    # Check containment (includes boundary by default)
    mask = polygon_path.contains_points(points)
    
    if exclude_boundary:
        # Check if points are on the boundary using contains_points with radius
        # Points on boundary will be excluded
        boundary_mask = polygon_path.contains_points(points, radius=-boundary_tolerance)
        
        # Keep only points that are strictly inside (not on boundary)
        mask = boundary_mask
    
    return mask.reshape(x.shape)


def compute_geometry_mask(outer_curve_x, outer_curve_y, inner_curves_x=None, inner_curves_y=None, 
                         square_bounds=(-1.5, 1.5, -1.5, 1.5), pixel_res=64, exclude_boundary=True):
    """
    Compute a geometry mask on a square domain.
    
    Parameters:
    - outer_curve_x, outer_curve_y: arrays defining the outer boundary curve
    - inner_curves_x, inner_curves_y: optional list of arrays defining inner boundary curves (holes)
    - square_bounds: tuple (x_min, x_max, y_min, y_max) defining the square domain
    - pixel_res: number of pixels along each dimension (creates pixel_res x pixel_res grid)
    - exclude_boundary: if True, points on boundaries are excluded from the domain
    
    Returns:
    - mask: 2D array of shape (pixel_res, pixel_res) with values 1 inside domain, 0 outside
    - x_grid, y_grid: 2D arrays with the x,y coordinates of each pixel center
    """
    x_min, x_max, y_min, y_max = square_bounds
    
    # Create pixel grid
    x_edges = np.linspace(x_min, x_max, pixel_res + 1)
    y_edges = np.linspace(y_min, y_max, pixel_res + 1)
    
    # Pixel centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    # Create 2D grid of pixel centers
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='xy')
    
    # Initialize mask (all pixels outside by default)
    mask = np.zeros((pixel_res, pixel_res), dtype=int)
    
    # Check if pixels are inside outer boundary
    if outer_curve_x is not None and outer_curve_y is not None:
        inside_outer = point_in_polygon(x_grid, y_grid, outer_curve_x, outer_curve_y, exclude_boundary)
        mask = inside_outer.astype(int)
    
    # Remove pixels inside inner boundaries (holes)
    if inner_curves_x is not None and inner_curves_y is not None:
        if not isinstance(inner_curves_x, list):
            inner_curves_x = [inner_curves_x]
            inner_curves_y = [inner_curves_y]
        
        for inner_x, inner_y in zip(inner_curves_x, inner_curves_y):
            inside_inner = point_in_polygon(x_grid, y_grid, inner_x, inner_y, exclude_boundary)
            mask = mask & (~inside_inner)  # Remove inner regions
    
    return mask, x_grid, y_grid


def visualize_geometry_mask(mask, x_grid, y_grid, outer_curve_x=None, outer_curve_y=None, 
                           inner_curve_x=None, inner_curve_y=None, title="Geometry Mask"):
    """
    Visualize the geometry mask and boundary curves.
    
    Parameters:
    - mask: 2D array representing the geometry mask
    - x_grid, y_grid: 2D arrays with pixel coordinates
    - outer_curve_x, outer_curve_y: outer boundary curve points (optional)
    - inner_curve_x, inner_curve_y: inner boundary curve points (optional)
    - title: plot title
    
    Returns:
    - fig, ax: matplotlib figure and axis objects
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Display mask
        extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
        im = ax.imshow(mask, extent=extent, origin='lower', cmap='seismic', 
                      alpha=0.8, interpolation='nearest')
        
        # Plot boundary curves
        if outer_curve_x is not None and outer_curve_y is not None:
            ax.plot(outer_curve_x, outer_curve_y, 'r-', linewidth=2, label='Outer boundary Г1')
        
        if inner_curve_x is not None and inner_curve_y is not None:
            ax.plot(inner_curve_x, inner_curve_y, 'b-', linewidth=2, label='Inner boundary Г2')

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        
        if outer_curve_x is not None or inner_curve_x is not None:
            ax.legend()
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Mask value')
        
        plt.tight_layout()
        return fig, ax
        
    except ImportError:
        print("matplotlib not available - skipping visualization")
        return None, None

# endregion Geometry mask

# region Boundary mask

def point_on_boundary(x, y, poly_x, poly_y, boundary_tolerance=1e-6):
    """
    Determine if points (x,y) are on the boundary of a polygon defined by vertices (poly_x, poly_y).
    
    Parameters:
    - x, y: arrays of query points
    - poly_x, poly_y: arrays defining the polygon vertices
    - boundary_tolerance: tolerance for boundary detection (distance from boundary)
    
    Returns:
    - mask: boolean array, True if point is on the boundary
    """
    # Create polygon path
    vertices = np.column_stack((poly_x, poly_y))
    polygon_path = Path(vertices)
    
    # Query points
    points = np.column_stack((x.flatten(), y.flatten()))
    
    # Check if points are inside the polygon with positive tolerance (slightly expanded)
    inside_expanded = polygon_path.contains_points(points, radius=boundary_tolerance)
    
    # Check if points are inside the polygon with negative tolerance (slightly contracted)
    inside_contracted = polygon_path.contains_points(points, radius=-boundary_tolerance)
    
    # Points on boundary are inside expanded but outside contracted
    on_boundary = inside_expanded & (~inside_contracted)
    
    return on_boundary.reshape(x.shape)


def compute_boundary_mask(outer_curve_x, outer_curve_y, inner_curves_x=None, inner_curves_y=None,
                         square_bounds=(-1.5, 1.5, -1.5, 1.5), pixel_res=64, boundary_tolerance=None):
    """
    Compute a boundary mask on a square domain.
    
    Parameters:
    - outer_curve_x, outer_curve_y: arrays defining the outer boundary curve
    - inner_curves_x, inner_curves_y: optional list of arrays defining inner boundary curves
    - square_bounds: tuple (x_min, x_max, y_min, y_max) defining the square domain
    - pixel_res: number of pixels along each dimension (creates pixel_res x pixel_res grid)
    - boundary_tolerance: tolerance for boundary detection (default: adaptive based on grid resolution)
    
    Returns:
    - boundary_mask: 2D array of shape (pixel_res, pixel_res) with values 1 on boundary, 0 elsewhere
    - x_grid, y_grid: 2D arrays with the x,y coordinates of each pixel center
    """
    x_min, x_max, y_min, y_max = square_bounds
    
    # Create pixel grid
    x_edges = np.linspace(x_min, x_max, pixel_res + 1)
    y_edges = np.linspace(y_min, y_max, pixel_res + 1)
    
    # Pixel centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    # Create 2D grid of pixel centers
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='xy')
    
    # Adaptive boundary tolerance based on grid resolution
    if boundary_tolerance is None:
        pixel_size = min((x_max - x_min) / pixel_res, (y_max - y_min) / pixel_res)
        boundary_tolerance = pixel_size / 2  # Half pixel size
    
    # Initialize boundary mask (all pixels not on boundary by default)
    boundary_mask = np.zeros((pixel_res, pixel_res), dtype=int)
    
    # Check if pixels are on outer boundary
    if outer_curve_x is not None and outer_curve_y is not None:
        on_outer_boundary = point_on_boundary(x_grid, y_grid, outer_curve_x, outer_curve_y, boundary_tolerance)
        boundary_mask = on_outer_boundary.astype(int)
    
    # Add pixels on inner boundaries
    if inner_curves_x is not None and inner_curves_y is not None:
        if not isinstance(inner_curves_x, list):
            inner_curves_x = [inner_curves_x]
            inner_curves_y = [inner_curves_y]
        
        for inner_x, inner_y in zip(inner_curves_x, inner_curves_y):
            on_inner_boundary = point_on_boundary(x_grid, y_grid, inner_x, inner_y, boundary_tolerance)
            boundary_mask = boundary_mask | on_inner_boundary.astype(int)  # Union of boundaries
    
    return boundary_mask, x_grid, y_grid


def visualize_boundary_mask(boundary_mask, x_grid, y_grid, title="Boundary Mask"):
    """
    Visualize the boundary mask as red dots on white background.
    
    Parameters:
    - boundary_mask: 2D array representing the boundary mask
    - x_grid, y_grid: 2D arrays with pixel coordinates
    - title: plot title
    
    Returns:
    - fig, ax: matplotlib figure and axis objects
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Set white background
        ax.set_facecolor('white')
        
        # Find boundary pixel coordinates
        boundary_indices = np.where(boundary_mask == 1)
        if len(boundary_indices[0]) > 0:
            # Get actual coordinates of boundary pixels
            boundary_x = x_grid[boundary_indices]
            boundary_y = y_grid[boundary_indices]
            
            # Plot boundary pixels as red dots
            ax.scatter(boundary_x, boundary_y, c='red', s=20, alpha=0.8, 
                      marker='s', label=f'Boundary pixels ({len(boundary_x)})')
        
        # Set plot limits based on grid
        ax.set_xlim(x_grid.min(), x_grid.max())
        ax.set_ylim(y_grid.min(), y_grid.max())
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        
        # Show legend if we have boundary pixels
        if len(boundary_indices[0]) > 0:
            ax.legend()
        
        plt.tight_layout()
        return fig, ax
        
    except ImportError:
        print("matplotlib not available - skipping visualization")
        return None, None

# endregion Boundary mask

# region Boundary condition visualization

def visualize_dirichlet_mask(dirichlet_mask, x_grid, y_grid, boundary_mask=None, title="Dirichlet Boundary Conditions"):
    """
    Visualize the Dirichlet boundary mask with values shown as a colormap.
    
    Parameters:
    - dirichlet_mask: 2D array with Dirichlet values (0 for non-boundary, values for boundary)
    - x_grid, y_grid: 2D arrays with pixel coordinates
    - boundary_mask: optional 2D array indicating boundary locations (if provided, used instead of dirichlet_mask != 0)
    - title: plot title
    
    Returns:
    - fig, ax: matplotlib figure and axis objects
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Determine boundary indices
        if boundary_mask is not None:
            boundary_indices = np.where(boundary_mask == 1)
        else:
            boundary_indices = np.where(dirichlet_mask != 0)
        
        if len(boundary_indices[0]) > 0:
            # Get actual coordinates and values of boundary pixels
            boundary_x = x_grid[boundary_indices]
            boundary_y = y_grid[boundary_indices]
            boundary_values = dirichlet_mask[boundary_indices]
            
            # Create scatter plot with color representing the Dirichlet values
            scatter = ax.scatter(boundary_x, boundary_y, c=boundary_values, 
                               s=30, alpha=0.8, marker='s', cmap='viridis')
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='u(x,y) Dirichlet values')
            
            ax.text(0.02, 0.98, f'Boundary points: {len(boundary_x)}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No Dirichlet boundary conditions found', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Set plot limits based on grid
        ax.set_xlim(x_grid.min(), x_grid.max())
        ax.set_ylim(y_grid.min(), y_grid.max())
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig, ax
        
    except ImportError:
        print("matplotlib not available - skipping visualization")
        return None, None


def visualize_neumann_mask(neumann_mask, x_grid, y_grid, boundary_mask=None, title="Neumann Boundary Conditions"):
    """
    Visualize the Neumann boundary mask with normal derivative values shown as a colormap.
    
    Parameters:
    - neumann_mask: 2D array with Neumann values (0 for non-boundary, du/dn values for boundary)
    - x_grid, y_grid: 2D arrays with pixel coordinates
    - boundary_mask: optional 2D array indicating boundary locations (if provided, used instead of neumann_mask != 0)
    - title: plot title
    
    Returns:
    - fig, ax: matplotlib figure and axis objects
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Determine boundary indices
        if boundary_mask is not None:
            boundary_indices = np.where(boundary_mask == 1)
        else:
            boundary_indices = np.where(neumann_mask != 0)
        
        if len(boundary_indices[0]) > 0:
            # Get actual coordinates and values of boundary pixels
            boundary_x = x_grid[boundary_indices]
            boundary_y = y_grid[boundary_indices]
            boundary_values = neumann_mask[boundary_indices]
            
            # Create scatter plot with color representing the Neumann values
            scatter = ax.scatter(boundary_x, boundary_y, c=boundary_values, 
                               s=30, alpha=0.8, marker='s', cmap='RdBu_r')
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='∂u/∂n Neumann values')
            
            ax.text(0.02, 0.98, f'Boundary points: {len(boundary_x)}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No Neumann boundary conditions found', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Set plot limits based on grid
        ax.set_xlim(x_grid.min(), x_grid.max())
        ax.set_ylim(y_grid.min(), y_grid.max())
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig, ax
        
    except ImportError:
        print("matplotlib not available - skipping visualization")
        return None, None


def visualize_boundary_conditions_combined(dirichlet_mask, neumann_mask, x_grid, y_grid, 
                                          boundary_mask=None, title="Combined Boundary Conditions"):
    """
    Visualize both Dirichlet and Neumann boundary conditions in a single plot.
    
    Parameters:
    - dirichlet_mask: 2D array with Dirichlet values
    - neumann_mask: 2D array with Neumann values
    - x_grid, y_grid: 2D arrays with pixel coordinates
    - boundary_mask: optional 2D array indicating boundary locations
    - title: plot title
    
    Returns:
    - fig, ax: matplotlib figure and axis objects
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Determine boundary indices
        if boundary_mask is not None:
            boundary_indices = np.where(boundary_mask == 1)
        else:
            dirichlet_indices = np.where(dirichlet_mask != 0)
            neumann_indices = np.where(neumann_mask != 0)
        
        # Plot Dirichlet conditions
        if boundary_mask is not None:
            dirichlet_indices = boundary_indices
        else:
            dirichlet_indices = np.where(dirichlet_mask != 0)
            
        if len(dirichlet_indices[0]) > 0:
            boundary_x = x_grid[dirichlet_indices]
            boundary_y = y_grid[dirichlet_indices]
            boundary_values = dirichlet_mask[dirichlet_indices]
            
            scatter1 = ax1.scatter(boundary_x, boundary_y, c=boundary_values, 
                                 s=30, alpha=0.8, marker='s', cmap='viridis')
            plt.colorbar(scatter1, ax=ax1, label='u(x,y)')
            
            ax1.text(0.02, 0.98, f'Points: {len(boundary_x)}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_xlim(x_grid.min(), x_grid.max())
        ax1.set_ylim(y_grid.min(), y_grid.max())
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Dirichlet Conditions')
        
        # Plot Neumann conditions
        if boundary_mask is not None:
            neumann_indices = boundary_indices
        else:
            neumann_indices = np.where(neumann_mask != 0)
            
        if len(neumann_indices[0]) > 0:
            boundary_x = x_grid[neumann_indices]
            boundary_y = y_grid[neumann_indices]
            boundary_values = neumann_mask[neumann_indices]
            
            scatter2 = ax2.scatter(boundary_x, boundary_y, c=boundary_values, 
                                 s=30, alpha=0.8, marker='s', cmap='RdBu_r')
            plt.colorbar(scatter2, ax=ax2, label='∂u/∂n')
            
            ax2.text(0.02, 0.98, f'Points: {len(boundary_x)}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlim(x_grid.min(), x_grid.max())
        ax2.set_ylim(y_grid.min(), y_grid.max())
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Neumann Conditions')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig, (ax1, ax2)
        
    except ImportError:
        print("matplotlib not available - skipping visualization")
        return None, None

# endregion Boundary condition visualization

# Visualize the true solution field on the interior
def visualize_interior_solution(u_mask, x_grid, y_grid, geometry_mask=None, title="Interior Solution"):
    """
    Visualize the solution field on interior points.
    
    Parameters:
    - u_mask: 2D array with solution values (0 for non-interior, values for interior)
    - x_grid, y_grid: 2D arrays with pixel coordinates
    - geometry_mask: optional 2D array indicating interior locations
    - title: plot title
    
    Returns:
    - fig, ax: matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Determine interior indices
    if geometry_mask is not None:
        interior_indices = np.where(geometry_mask == 1)
    else:
        interior_indices = np.where(u_mask != 0)
    
    if len(interior_indices[0]) > 0:
        # Get actual coordinates and values of interior pixels
        interior_x = x_grid[interior_indices]
        interior_y = y_grid[interior_indices]
        interior_values = u_mask[interior_indices]
        
        # Create scatter plot with color representing the solution values
        scatter = ax.scatter(interior_x, interior_y, c=interior_values, 
                           s=20, alpha=0.7, marker='s', cmap='RdBu_r')
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('u(x,y)', rotation=270, labelpad=20)
        
        # Set equal aspect ratio and add grid
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        
        # Set reasonable axis limits
        x_margin = 0.1 * (np.max(interior_x) - np.min(interior_x))
        y_margin = 0.1 * (np.max(interior_y) - np.min(interior_y))
        ax.set_xlim(np.min(interior_x) - x_margin, np.max(interior_x) + x_margin)
        ax.set_ylim(np.min(interior_y) - y_margin, np.max(interior_y) + y_margin)
    else:
        ax.text(0.5, 0.5, 'No interior points found', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
    
    return fig, ax


def create_coordinate_grids(H, W, grid_size=3.0, device=None):
    """Create coordinate grids at pixel centers for [-grid_size/2, grid_size/2]^2."""
    dx = grid_size / W
    dy = grid_size / H
    x = linspace(-grid_size/2 + dx/2, grid_size/2 - dx/2, W, device=device)  # centers in x
    y = linspace(-grid_size/2 + dy/2, grid_size/2 - dy/2, H, device=device)  # centers in y
    Y, X = meshgrid(y, x, indexing='ij')  # [H, W] - Y[i,j]=y[i], X[i,j]=x[j]
    return X.unsqueeze(0).unsqueeze(0), Y.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
