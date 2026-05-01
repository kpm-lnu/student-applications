import torch
import pytest
from harmonic.kernels import get_neumann_kernel


class TestGetNeumannKernel:
    """Test suite for the get_neumann_kernel function."""
    
    def test_return_structure(self):
        """Test that function returns tuple of two tensors."""
        result = get_neumann_kernel(h=0.2)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        dx, dy = result
        assert isinstance(dx, torch.Tensor)
        assert isinstance(dy, torch.Tensor)
    
    def test_kernel_shapes(self):
        """Test that both kernels have correct shape [1,1,3,3]."""
        dx, dy = get_neumann_kernel(h=0.2)
        
        assert dx.shape == (1, 1, 3, 3)
        assert dy.shape == (1, 1, 3, 3)
    
    def test_dx_kernel_values(self):
        """Test that dx kernel has correct forward difference pattern."""
        h = 0.2
        dx, _ = get_neumann_kernel(h=h)
        
        # Remove batch and channel dimensions for easier testing
        dx_2d = dx.squeeze()
        
        scale = 1.0 / (2.0 * h)
        expected = torch.tensor(
            [[0., 0., 0.],
             [-scale, 0, scale],
             [0., 0., 0.]],
            dtype=torch.float32
        )
        
        torch.testing.assert_close(dx_2d, expected, rtol=1e-6, atol=1e-6)
    
    def test_dy_kernel_values(self):
        """Test that dy kernel has correct forward difference pattern."""
        h = 0.2
        _, dy = get_neumann_kernel(h=h)
        
        # Remove batch and channel dimensions for easier testing
        dy_2d = dy.squeeze()
        
        scale = 1.0 / (2.0 * h)
        expected = torch.tensor(
            [[0., -scale, 0.],
             [0.,  0., 0.],
             [0.,  scale, 0.]],
            dtype=torch.float32
        )
        
        torch.testing.assert_close(dy_2d, expected, rtol=1e-6, atol=1e-6)
    

    def test_device_placement(self):
        """Test that kernels are placed on specified device."""
        
        # Test CPU placement (default)
        dx_cpu, dy_cpu = get_neumann_kernel(h=0.2, device='cpu')
        assert dx_cpu.device.type == 'cpu'
        assert dy_cpu.device.type == 'cpu'
        
        # Test explicit None device
        dx_none, dy_none = get_neumann_kernel(h=0.2, device=None)
        assert dx_none.device.type == 'cpu'
        assert dy_none.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_placement(self):
        """Test CUDA device placement if available."""
        dx_cuda, dy_cuda = get_neumann_kernel(h=0.2, device='cuda')
        dx_cuda, dy_cuda = get_neumann_kernel(h=0.2, device='cuda')
        
        assert dx_cuda.device.type == 'cuda'
        assert dy_cuda.device.type == 'cuda'
    
    def test_dtype_consistency(self):
        """Test that kernels have correct dtype."""
        dx, dy = get_neumann_kernel(h=0.2)
        
        assert dx.dtype == torch.float32
        assert dy.dtype == torch.float32
    
    def test_mathematical_properties(self):
        """Test mathematical properties of the kernels."""
        dx, dy = get_neumann_kernel(h=0.2)
        
        # Sum of dx kernel should be 0 (finite difference property)
        dx_sum = dx.sum().item()
        assert abs(dx_sum) < 1e-6
        
        # Sum of dy kernel should be 0 (finite difference property)
        dy_sum = dy.sum().item()
        assert abs(dy_sum) < 1e-6
    

    def test_zero_elements_positions(self):
        """Test that zero elements are in correct positions."""
        dx, dy = get_neumann_kernel(h=0.2)
        
        dx_2d = dx.squeeze()
        assert dx_2d[0, 0] == 0 and dx_2d[0, 1] == 0 and dx_2d[0, 2] == 0
        assert dx_2d[2, 0] == 0 and dx_2d[2, 1] == 0 and dx_2d[2, 2] == 0
        assert dx_2d[1, 1] == 0
        
        # dy kernel: zeros everywhere except middle column
        dy_2d = dy.squeeze()
        assert dy_2d[0, 0] == 0 and dy_2d[0, 2] == 0
        assert dy_2d[1, 0] == 0 and dy_2d[1, 1] == 0 and dy_2d[1, 2] == 0
        assert dy_2d[2, 0] == 0 and dy_2d[2, 2] == 0