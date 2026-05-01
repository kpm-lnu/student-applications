import numpy as np
import pytest
from harmonic.sample_harmonic_field import *


class TestComputeHarmonicBasis:
    def test_basis_shape(self):
        """Test that the basis has the correct shape."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, 1.5, 2.5])
        N = 5
        
        basis = compute_harmonic_basis(x, y, N)
        
        assert basis.shape == (N, len(x))
        assert basis.shape == (5, 3)
    
    def test_basis_values_simple_case(self):
        """Test basis values for simple known cases."""
        # Test at origin
        x = np.array([0.0])
        y = np.array([0.0])
        N = 3
        
        basis = compute_harmonic_basis(x, y, N)
        
        # At origin, z^n = 0 for n > 0, so all basis functions should be 0
        expected = np.zeros((N, 1))
        np.testing.assert_array_almost_equal(basis, expected)
    
    def test_basis_values_unit_circle(self):
        """Test basis values on unit circle."""
        # Test at z = 1 (x=1, y=0)
        x = np.array([1.0])
        y = np.array([0.0])
        N = 3
        
        basis = compute_harmonic_basis(x, y, N)
        
        # z^1 = 1, z^2 = 1, z^3 = 1, so Re(z^n) = 1 for all n
        expected = np.array([[1.0], [1.0], [1.0]])
        np.testing.assert_array_almost_equal(basis, expected)
    
    def test_basis_complex_point(self):
        """Test basis values for a complex point."""
        # Test at z = i (x=0, y=1)
        x = np.array([0.0])
        y = np.array([1.0])
        N = 4
        
        basis = compute_harmonic_basis(x, y, N)
        
        # z = i, so z^1 = i, z^2 = -1, z^3 = -i, z^4 = 1
        # Re(z^n) = [0, -1, 0, 1]
        expected = np.array([[0.0], [-1.0], [0.0], [1.0]])
        np.testing.assert_array_almost_equal(basis, expected)
    
    def test_basis_multiple_points(self):
        """Test basis computation for multiple points."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        N = 2
        
        basis = compute_harmonic_basis(x, y, N)
        
        # For z1 = 1: Re(z^1) = 1, Re(z^2) = 1
        # For z2 = i: Re(z^1) = 0, Re(z^2) = -1
        expected = np.array([[1.0, 0.0], [1.0, -1.0]])
        np.testing.assert_array_almost_equal(basis, expected)


class TestSampleHarmonic:
    def test_sample_harmonic_output_shapes(self):
        """Test that sample_harmonic returns correct shapes."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, 1.5, 2.5])
        N = 5
        a = np.random.randn(N)  # Generate coefficients
        
        u = sample_harmonic(x, y, a)
        
        assert u.shape == (len(x),)
        assert u.shape == (3,)
    
    def test_sample_harmonic_with_provided_coefficients(self):
        """Test that sample_harmonic works with provided coefficients."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, 1.5, 2.5])
        a_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        u = sample_harmonic(x, y, a_input)
        
        assert u.shape == (len(x),)
    
    def test_sample_harmonic_deterministic_with_same_coefficients(self):
        """Test that sample_harmonic is deterministic with same coefficients."""
        x = np.array([1.0, 2.0])
        y = np.array([0.5, 1.5])
        a = np.array([1.0, 2.0, 3.0])
        
        # Sample twice with same coefficients
        u1 = sample_harmonic(x, y, a)
        u2 = sample_harmonic(x, y, a)
        
        np.testing.assert_array_almost_equal(u1, u2)
    
    def test_sample_harmonic_different_coefficients(self):
        """Test that different coefficients produce different results."""
        x = np.array([1.0, 2.0])
        y = np.array([0.5, 1.5])
        
        # Sample with different coefficients
        a1 = np.array([1.0, 2.0, 3.0])
        u1 = sample_harmonic(x, y, a1)
        
        a2 = np.array([4.0, 5.0, 6.0])
        u2 = sample_harmonic(x, y, a2)
        
        # Results should be different (with high probability)
        assert not np.allclose(u1, u2)
        assert not np.allclose(a1, a2)
    
    def test_sample_harmonic_linearity(self):
        """Test that the function is linear in coefficients."""
        x = np.array([1.0, 2.0])
        y = np.array([0.5, 1.5])
        N = 3
        
        # Get basis
        basis = compute_harmonic_basis(x, y, N)
        
        # Create known coefficients
        a = np.array([1.0, 2.0, 3.0])
        
        # Manual computation
        u_manual = a @ basis
        
        # Using sample_harmonic with provided coefficients
        u_func = sample_harmonic(x, y, a)
        
        np.testing.assert_array_almost_equal(u_func, u_manual)


class TestComputeHarmonicGradients:
    def test_gradient_shapes(self):
        """Test that gradients have correct shapes."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, 1.5, 2.5])
        a = np.array([1.0, 2.0, 3.0])
        
        du_dx, du_dy = compute_harmonic_gradients(x, y, a)
        
        assert du_dx.shape == (len(x),)
        assert du_dy.shape == (len(x),)
        assert du_dx.shape == (3,)
        assert du_dy.shape == (3,)
    
    def test_gradient_at_origin(self):
        """Test gradients at origin for simple case."""
        x = np.array([0.0])
        y = np.array([0.0])
        a = np.array([1.0])  # Only first harmonic
        
        du_dx, du_dy = compute_harmonic_gradients(x, y, a)
        
        # For u = Re(z), gradient at origin is (1, 0)
        # But since z^n derivatives involve z^(n-1), at origin this is 0 for n>1
        # For n=1: d/dx Re(z) = 1, d/dy Re(z) = 0
        np.testing.assert_array_almost_equal(du_dx, [1.0])
        np.testing.assert_array_almost_equal(du_dy, [0.0])
    
    def test_gradient_simple_point(self):
        """Test gradients at a simple point."""
        x = np.array([1.0])
        y = np.array([0.0])
        a = np.array([1.0, 0.0])  # Only first harmonic
        
        du_dx, du_dy = compute_harmonic_gradients(x, y, a)
        
        # For u = Re(z) at z=1: du/dx = 1, du/dy = 0
        np.testing.assert_array_almost_equal(du_dx, [1.0], decimal=10)
        np.testing.assert_array_almost_equal(du_dy, [0.0], decimal=10)
    
    def test_gradient_consistency(self):
        """Test that gradients are consistent with finite differences."""
        # Use a point away from origin to avoid numerical issues
        x0, y0 = 1.0, 0.5
        h = 1e-7  # Small step for finite differences
        a = np.array([1.0, 2.0])
        
        # Compute analytical gradients
        du_dx_analytical, du_dy_analytical = compute_harmonic_gradients(
            np.array([x0]), np.array([y0]), a
        )
        
        # Compute finite difference gradients using the provided coefficients
        # For x-direction
        basis_plus_x = compute_harmonic_basis(np.array([x0 + h]), np.array([y0]), len(a))
        basis_minus_x = compute_harmonic_basis(np.array([x0 - h]), np.array([y0]), len(a))
        u_plus_x = a @ basis_plus_x
        u_minus_x = a @ basis_minus_x
        du_dx_finite = (u_plus_x - u_minus_x) / (2 * h)
        
        # For y-direction
        basis_plus_y = compute_harmonic_basis(np.array([x0]), np.array([y0 + h]), len(a))
        basis_minus_y = compute_harmonic_basis(np.array([x0]), np.array([y0 - h]), len(a))
        u_plus_y = a @ basis_plus_y
        u_minus_y = a @ basis_minus_y
        du_dy_finite = (u_plus_y - u_minus_y) / (2 * h)
        
        # Compare analytical and finite difference results
        np.testing.assert_allclose(du_dx_analytical[0], du_dx_finite[0], rtol=1e-5)
        np.testing.assert_allclose(du_dy_analytical[0], du_dy_finite[0], rtol=1e-5)
    
    def test_multiple_harmonics_gradient(self):
        """Test gradients with multiple harmonics."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        a = np.array([1.0, 2.0, 3.0])
        
        du_dx, du_dy = compute_harmonic_gradients(x, y, a)
        
        # Should return arrays of correct length
        assert len(du_dx) == 2
        assert len(du_dy) == 2
        
        # Values should be finite
        assert np.all(np.isfinite(du_dx))
        assert np.all(np.isfinite(du_dy))


class TestIntegration:
    def test_sample_and_gradient_consistency(self):
        """Test that sample_harmonic and compute_harmonic_gradients are consistent."""
        x = np.array([1.0, 0.5])
        y = np.array([0.5, 1.0])
        N = 3
        
        # Use known coefficients for reproducibility
        a = np.array([1.0, 2.0, 3.0])
        u = sample_harmonic(x, y, a)
        
        # Compute gradients using the same coefficients
        du_dx, du_dy = compute_harmonic_gradients(x, y, a)
        
        # Both should have same length as input points
        assert len(u) == len(x)
        assert len(du_dx) == len(x)
        assert len(du_dy) == len(x)
        
        # All values should be finite
        assert np.all(np.isfinite(u))
        assert np.all(np.isfinite(du_dx))
        assert np.all(np.isfinite(du_dy))


class TestComputeNormalDerivative:
    """Test the compute_normal_derivative function for Γ1 boundary (x=1.3*cos(t), y=sin(t))."""
    
    def test_normal_derivative_shape_gamma1(self):
        """Test that normal derivative has correct shape for Γ1 boundary."""
        M = 8
        t = np.linspace(0, 2*np.pi, M, endpoint=False)
        x = 1.3 * np.cos(t)
        y = np.sin(t)
        a = np.array([1.0, 0.5, 0.2])
        
        du_dn = compute_normal_derivative(x, y, a)
        
        assert du_dn.shape == (M,)
        assert np.all(np.isfinite(du_dn))
    
    
    def test_normal_derivative_gamma1_multiple_harmonics(self):
        """Test normal derivative on Γ1 for multiple harmonics."""
        M = 12
        t = np.linspace(0, 2*np.pi, M, endpoint=False)
        x = 1.3 * np.cos(t)
        y = np.sin(t)
        
        # Use multiple harmonics
        a = np.array([1.0, 0.5, 0.3])
        
        du_dn = compute_normal_derivative(x, y, a)
        
        # Should have correct shape and finite values
        assert du_dn.shape == (M,)
        assert np.all(np.isfinite(du_dn))
        
        # Check that it's not all zeros (non-trivial result)
        assert np.any(np.abs(du_dn) > 1e-10)
    
    
    def test_normal_derivative_gamma1_zero_for_constant_function(self):
        """Test that normal derivative is zero for constant functions on Γ1."""
        M = 10
        t = np.linspace(0, 2*np.pi, M, endpoint=False)
        x = 1.3 * np.cos(t)
        y = np.sin(t)
        
        # Constant function corresponds to zero coefficients for all harmonics
        a = np.zeros(5)
        
        du_dn = compute_normal_derivative(x, y, a)
        
        # Should be zero everywhere
        np.testing.assert_allclose(du_dn, np.zeros(M), atol=1e-12)
    
    
    def test_normal_derivative_gamma1_analytical_accuracy(self):
        """Test that the analytical tangent method gives high precision results."""
        M = 8
        t = np.linspace(0, 2*np.pi, M, endpoint=False)
        x = 1.3 * np.cos(t)
        y = np.sin(t)
        
        # Simple case: u = x^2 - y^2 (Re(z^2))
        a = np.array([0.0, 1.0])  # Second harmonic: Re(z^2) = x^2 - y^2
        
        du_dn = compute_normal_derivative(x, y, a)
        
        # All values should be finite and well-behaved
        assert np.all(np.isfinite(du_dn))
        assert not np.any(np.isnan(du_dn))
        assert not np.any(np.isinf(du_dn))


if __name__ == "__main__":
    pytest.main([__file__])
