import numpy as np
import pytest
from compute_grid import (
    point_on_boundary, 
    compute_boundary_mask,
)
from compute_grid import compute_geometry_mask


class TestPointOnBoundary:
    def test_simple_square_boundary(self):
        """Test boundary detection for a simple square."""
        # Define a unit square
        poly_x = np.array([0, 1, 1, 0, 0])
        poly_y = np.array([0, 0, 1, 1, 0])
        
        # Test points: inside, outside, on boundary
        x_test = np.array([0.5, 1.5, 0.0, 1.0, 0.5])  # center, outside, corner, corner, edge
        y_test = np.array([0.5, 0.5, 0.0, 1.0, 0.0])  # center, outside, corner, corner, edge
        
        boundary_mask = point_on_boundary(x_test, y_test, poly_x, poly_y, boundary_tolerance=0.01)
        
        # Center should not be on boundary, outside should not be on boundary
        assert boundary_mask[0] == False  # Center point
        assert boundary_mask[1] == False  # Outside point
        # Boundary points should be detected (corners and edges)
        assert boundary_mask[2] == True
        assert boundary_mask[3] == True
        assert boundary_mask[4] == True

    def test_circle_boundary(self):
        """Test boundary detection for a circle."""
        # Create circle points
        n_points = 100
        t = np.linspace(0, 2*np.pi, n_points, endpoint=True)
        radius = 1.0
        poly_x = radius * np.cos(t)
        poly_y = radius * np.sin(t)
        
        # Test points
        x_test = np.array([0.0, 2.0, 1.0, 0.5])  # center, outside, on circle, inside
        y_test = np.array([0.0, 0.0, 0.0, 0.0])  # center, outside, on circle, inside
        
        boundary_mask = point_on_boundary(x_test, y_test, poly_x, poly_y, boundary_tolerance=0.05)
        
        assert boundary_mask[0] == False  # Center
        assert boundary_mask[1] == False  # Outside
        assert boundary_mask[2] == True   # On circle
        assert boundary_mask[3] == False  # Inside
    
    def test_tolerance_effect(self):
        """Test that boundary tolerance affects detection."""
        # Simple square
        poly_x = np.array([0, 1, 1, 0, 0])
        poly_y = np.array([0, 0, 1, 1, 0])
        
        # Point slightly inside the boundary
        x_test = np.array([0.01])
        y_test = np.array([0.5])
        
        # With small tolerance, should not be detected as boundary
        boundary_small = point_on_boundary(x_test, y_test, poly_x, poly_y, boundary_tolerance=0.005)
        # With large tolerance, should be detected as boundary
        boundary_large = point_on_boundary(x_test, y_test, poly_x, poly_y, boundary_tolerance=0.02)
        
        assert boundary_small[0] == False
        assert boundary_large[0] == True


class TestComputeBoundaryMask:
    def test_boundary_mask_shape(self):
        """Test that boundary mask has correct shape."""
        # Simple square
        outer_x = np.array([0, 1, 1, 0, 0])
        outer_y = np.array([0, 0, 1, 1, 0])
        
        boundary_mask, x_grid, y_grid = compute_boundary_mask(
            outer_x, outer_y,
            square_bounds=(-0.5, 1.5, -0.5, 1.5),
            pixel_res=10
        )
        
        assert boundary_mask.shape == (10, 10)
        assert x_grid.shape == (10, 10)
        assert y_grid.shape == (10, 10)
        assert np.all((boundary_mask == 0) | (boundary_mask == 1))
    
    def test_boundary_mask_has_boundary_pixels(self):
        """Test that boundary mask contains some boundary pixels."""
        # Circle
        t = np.linspace(0, 2*np.pi, 64, endpoint=False)
        outer_x = np.cos(t)
        outer_y = np.sin(t)
        
        boundary_mask, _, _ = compute_boundary_mask(
            outer_x, outer_y,
            square_bounds=(-1.5, 1.5, -1.5, 1.5),
            pixel_res=32
        )
        
        # Should have some boundary pixels
        assert boundary_mask.sum() > 0
        # But not too many (shouldn't be most of the domain)
        assert boundary_mask.sum() < boundary_mask.size / 4
    
    def test_boundary_mask_with_hole(self):
        """Test boundary mask with inner boundary."""
        # Outer square
        outer_x = np.array([-1, 1, 1, -1, -1])
        outer_y = np.array([-1, -1, 1, 1, -1])
        
        # Inner square
        inner_x = np.array([-0.3, 0.3, 0.3, -0.3, -0.3])
        inner_y = np.array([-0.3, -0.3, 0.3, 0.3, -0.3])
        
        boundary_mask, _, _ = compute_boundary_mask(
            outer_x, outer_y, inner_x, inner_y,
            square_bounds=(-1.5, 1.5, -1.5, 1.5),
            pixel_res=20
        )
        
        # Should have more boundary pixels than without hole
        boundary_mask_no_hole, _, _ = compute_boundary_mask(
            outer_x, outer_y,
            square_bounds=(-1.5, 1.5, -1.5, 1.5),
            pixel_res=20
        )
        
        assert boundary_mask.sum() >= boundary_mask_no_hole.sum()




class TestIntegration:
    def test_geometry_mask_compatibility(self):
        """Test compatibility with __main__.py curve."""
        M = 64
        t = np.linspace(0, 2*np.pi, M, endpoint=False)
        x_curve = 1.3 * np.cos(t)
        y_curve = np.sin(t)
        
        # Compute both domain and boundary masks
        domain_mask, _, _ = compute_geometry_mask(
            x_curve, y_curve,
            square_bounds=(-1.5, 1.5, -1.5, 1.5),
            pixel_res=64
        )
        
        boundary_mask, _, _ = compute_boundary_mask(
            x_curve, y_curve,
            square_bounds=(-1.5, 1.5, -1.5, 1.5),
            pixel_res=64
        )
        
        # Verify basic properties
        assert domain_mask.shape == boundary_mask.shape
        assert domain_mask.sum() > boundary_mask.sum()
        assert boundary_mask.sum() > 0
        
        # Boundary pixels should be much fewer than domain pixels
        boundary_percentage = 100 * boundary_mask.sum() / boundary_mask.size
        domain_percentage = 100 * domain_mask.sum() / domain_mask.size
        
        assert boundary_percentage < domain_percentage
        assert boundary_percentage < 10  # Boundary should be < 10% of total pixels
