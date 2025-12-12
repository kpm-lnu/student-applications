import numpy as np
import pytest
from compute_grid import compute_geometry_mask, point_in_polygon


class TestPointInPolygon:
    def test_simple_square(self):
        """Test point-in-polygon with a simple square."""
        # Define a unit square
        poly_x = np.array([0, 1, 1, 0, 0])
        poly_y = np.array([0, 0, 1, 1, 0])
        
        # Test points
        x_test = np.array([0.5, 1.5, 0.0, 1.0])
        y_test = np.array([0.5, 0.5, 0.0, 1.0])
        
        mask = point_in_polygon(x_test, y_test, poly_x, poly_y)
        print(mask)
        
        # Point (0.5, 0.5) should be inside, (1.5, 0.5) should be outside
        # Corner points (0,0) and (1,1) might be on boundary
        expected = np.array([True, False, False, False])  # Conservative boundary treatment
        
        assert mask[0] == True  # Center point definitely inside
        assert mask[1] == False  # Point definitely outside
        assert mask[2] == False  # Point on the boundary excluded
        assert mask[3] == False  # Point on the boundary excluded

    def test_grid_points(self):
        """Test point-in-polygon with grid of points."""
        # Define a unit square
        poly_x = np.array([0, 1, 1, 0, 0])
        poly_y = np.array([0, 0, 1, 1, 0])
        
        # Create 3x3 grid
        x_grid = np.array([[0.25, 0.5, 0.75],
                          [0.25, 0.5, 0.75],
                          [0.25, 0.5, 0.75]])
        y_grid = np.array([[0.25, 0.25, 0.25],
                          [0.5,  0.5,  0.5],
                          [0.75, 0.75, 0.75]])
        
        mask = point_in_polygon(x_grid, y_grid, poly_x, poly_y)
        
        # All points should be inside the unit square
        assert mask.shape == (3, 3)
        assert np.all(mask)


class TestComputeGeometryMask:
    def test_simple_square_domain(self):
        """Test geometry mask computation with a simple square domain."""
        # Define outer boundary as a square
        outer_x = np.array([-0.5, 0.5, 0.5, -0.5, -0.5])
        outer_y = np.array([-0.5, -0.5, 0.5, 0.5, -0.5])
        
        mask, x_grid, y_grid = compute_geometry_mask(
            outer_x, outer_y,
            square_bounds=(-1.0, 1.0, -1.0, 1.0),
            pixel_res=10
        )
        
        # Check basic properties
        assert mask.shape == (10, 10)
        assert x_grid.shape == (10, 10)
        assert y_grid.shape == (10, 10)
        assert np.all((mask == 0) | (mask == 1))  # Only 0 and 1 values
        
        # Should have some pixels inside
        assert mask.sum() > 0
        assert mask.sum() < mask.size
    
    def test_no_outer_boundary(self):
        """Test with no outer boundary (should give all zeros)."""
        mask, x_grid, y_grid = compute_geometry_mask(
            None, None,
            square_bounds=(-1.0, 1.0, -1.0, 1.0),
            pixel_res=5
        )
        
        assert mask.shape == (5, 5)
        assert np.all(mask == 0)
    
    def test_with_inner_boundary(self):
        """Test geometry mask with inner boundary (hole)."""
        # Outer square
        outer_x = np.array([-1.0, 1.0, 1.0, -1.0, -1.0])
        outer_y = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
        
        # Inner square (hole)
        inner_x = np.array([-0.3, 0.3, 0.3, -0.3, -0.3])
        inner_y = np.array([-0.3, -0.3, 0.3, 0.3, -0.3])
        
        mask, x_grid, y_grid = compute_geometry_mask(
            outer_x, outer_y, inner_x, inner_y,
            square_bounds=(-1.5, 1.5, -1.5, 1.5),
            pixel_res=20
        )
        
        # Should have fewer inside pixels than without the hole
        mask_no_hole, _, _ = compute_geometry_mask(
            outer_x, outer_y, None, None,
            square_bounds=(-1.5, 1.5, -1.5, 1.5),
            pixel_res=20
        )
        
        assert mask.sum() < mask_no_hole.sum()
    
    def test_multiple_inner_boundaries(self):
        """Test with multiple inner boundaries."""
        # Large outer square
        outer_x = np.array([-1.0, 1.0, 1.0, -1.0, -1.0])
        outer_y = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
        
        # Two inner squares
        inner1_x = np.array([-0.6, -0.2, -0.2, -0.6, -0.6])
        inner1_y = np.array([-0.3, -0.3, 0.3, 0.3, -0.3])
        
        inner2_x = np.array([0.2, 0.6, 0.6, 0.2, 0.2])
        inner2_y = np.array([-0.3, -0.3, 0.3, 0.3, -0.3])
        
        mask, x_grid, y_grid = compute_geometry_mask(
            outer_x, outer_y, [inner1_x, inner2_x], [inner1_y, inner2_y],
            square_bounds=(-1.5, 1.5, -1.5, 1.5),
            pixel_res=20
        )
        
        # Should be less than with single hole
        mask_single_hole, _, _ = compute_geometry_mask(
            outer_x, outer_y, inner1_x, inner1_y,
            square_bounds=(-1.5, 1.5, -1.5, 1.5),
            pixel_res=20
        )
        
        assert mask.sum() <= mask_single_hole.sum()


class TestIntegration:
    def test_main_script_reproduction(self):
        """Test that we can reproduce the exact setup from __main__.py."""
        # Parameters from __main__.py
        M = 64
        t = np.linspace(0, 2*np.pi, M, endpoint=False)
        x_curve = 1.3 * np.cos(t)
        y_curve = np.sin(t)
        
        # Compute mask
        mask, x_grid, y_grid = compute_geometry_mask(
            outer_curve_x=x_curve,
            outer_curve_y=y_curve,
            square_bounds=(-1.5, 1.5, -1.5, 1.5),
            pixel_res=64
        )
        
        # Verify basic properties
        assert mask.shape == (64, 64)
        assert np.all((mask == 0) | (mask == 1))
        
        # Check approximate area
        percentage = 100 * mask.sum() / mask.size
        theoretical = 100 * np.pi * 1.3 * 1.0 / 9  # ≈ 45.4%
        
        assert abs(percentage - theoretical) < 1.0  # Within 1%
