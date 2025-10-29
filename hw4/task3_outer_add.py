import pytest
import torch
import triton
import triton.language as tl


def add_vec_torch(x, y):
    """Reference PyTorch implementation: outer sum of two vectors."""
    return x[None, :] + y[:, None]  # (N1, N0)


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, N0: tl.constexpr, N1: tl.constexpr, 
               BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr):
    """
    Triton kernel for outer sum of two vectors.
    
    Creates output[i, j] = x[j] + y[i] for all i, j.
    
    Args:
        x_ptr: Pointer to first vector (length N0)
        y_ptr: Pointer to second vector (length N1)
        output_ptr: Pointer to output matrix (N1, N0)
        N0: Size of first vector
        N1: Size of second vector
        BLOCK_SIZE_ROW: Block size for rows (B1 == N1)
        BLOCK_SIZE_COL: Block size for columns (B0 == N0)
    """
    # Get the program IDs for row and column blocks
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    
    # Compute starting positions
    row_start = row_pid * BLOCK_SIZE_ROW
    col_start = col_pid * BLOCK_SIZE_COL
    
    # Create range arrays for rows and columns
    row_indices = row_start + tl.arange(0, BLOCK_SIZE_ROW)
    col_indices = col_start + tl.arange(0, BLOCK_SIZE_COL)
    
    # Load data from vectors
    x_vals = tl.load(x_ptr + col_indices)  # Load x[col_start:col_start+BLOCK_SIZE_COL]
    y_vals = tl.load(y_ptr + row_indices)  # Load y[row_start:row_start+BLOCK_SIZE_ROW]
    
    # Create 2D blocks using broadcasting
    # x_2d: (1, BLOCK_SIZE_COL) -> broadcast to (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
    # y_2d: (BLOCK_SIZE_ROW, 1) -> broadcast to (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
    x_2d = x_vals[None, :]  # (1, BLOCK_SIZE_COL)
    y_2d = y_vals[:, None]  # (BLOCK_SIZE_ROW, 1)
    
    # Compute outer sum
    output = x_2d + y_2d  # Broadcasting to (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
    
    # Create 2D indices for the output matrix
    row_idx_2d = row_indices[:, None]  # (BLOCK_SIZE_ROW, 1)
    col_idx_2d = col_indices[None, :]  # (1, BLOCK_SIZE_COL)
    
    # Compute output offsets: row_idx * N0 + col_idx
    output_offsets = row_idx_2d * N0 + col_idx_2d  # (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
    
    # Store using tl.store which accepts 2D tensors
    tl.store(output_ptr + output_offsets, output)


def add_vec_triton(x, y):
    """
    Triton implementation of outer sum of two vectors.
    
    Args:
        x: First vector (length N0)
        y: Second vector (length N1)
        
    Returns:
        Output matrix of shape (N1, N0) where output[i, j] = x[j] + y[i]
    """
    # Ensure tensors are contiguous
    x = x.contiguous()
    y = y.contiguous()
    
    N0 = x.numel()
    N1 = y.numel()
    
    # Allocate output matrix
    output = torch.empty((N1, N0), dtype=x.dtype, device=x.device)
    
    # Choose block sizes (B0 == N0, B1 == N1)
    BLOCK_SIZE_ROW = N1
    BLOCK_SIZE_COL = N0
    
    # Grid size: (num_row_blocks, num_col_blocks)
    grid = (1, 1)
    
    # Launch kernel
    add_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        N0=N0,
        N1=N1,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )
    
    return output


@pytest.mark.parametrize("n0,n1", [
    (32, 16),
    (64, 32),
    (128, 128),
    (256, 512),
    (1024, 512),
    (32, 1024),
])
def test_outer_add(n0, n1):
    """Test Task 3: Outer Vector Add."""
    x = torch.randn(n0, dtype=torch.float32, device='cuda')
    y = torch.randn(n1, dtype=torch.float32, device='cuda')
    
    torch_result = add_vec_torch(x, y)
    triton_result = add_vec_triton(x, y)
    
    assert torch.allclose(torch_result, triton_result, rtol=1e-4, atol=1e-5), \
        f"Results don't match: n0={n0}, n1={n1}"


@pytest.mark.parametrize("n0,n1,dtype", [
    (64, 32, torch.float32),
    (128, 64, torch.float16),
])
def test_different_dtypes(n0, n1, dtype):
    """Test with different data types."""
    x = torch.randn(n0, dtype=dtype, device='cuda')
    y = torch.randn(n1, dtype=dtype, device='cuda')
    
    torch_result = add_vec_torch(x, y)
    triton_result = add_vec_triton(x, y)
    
    assert torch.allclose(torch_result, triton_result, rtol=1e-4, atol=1e-5), \
        f"Results don't match for dtype {dtype}"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Triton requires CUDA.")
    
    # Run tests with pytest from command line or run basic test
    pytest.main([__file__, "-v"])
