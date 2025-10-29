import pytest
import torch
import triton
import triton.language as tl


def add_torch(x, const_val):
    """Reference PyTorch implementation."""
    return x + const_val


@triton.jit
def add_kernel(x_ptr, const_val, output_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for adding a constant to a vector with masking.
    
    This is different from Task 1 in that B0 < N0, so we need to mask the tail.
    
    Args:
        x_ptr: Pointer to input tensor
        const_val: Constant value to add
        output_ptr: Pointer to output tensor
        N: Total number of elements
        BLOCK_SIZE: Number of elements processed by each program instance
    """
    # Get the program ID for this instance
    pid = tl.program_id(0)
    
    # Compute the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle cases where block extends beyond tensor size
    mask = offsets < N
    
    # Load data from global memory (masked)
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Add the constant
    output = x + const_val
    
    # Store the result back to global memory (masked)
    tl.store(output_ptr + offsets, output, mask=mask)


def add_triton(x, const_val):
    """
    Triton implementation of constant addition with masking support.
    
    Args:
        x: Input tensor
        const_val: Constant value to add
        
    Returns:
        Output tensor (x + const_val)
    """
    # Ensure tensor is contiguous
    x = x.contiguous()
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Get the size of the tensor
    n_elements = x.numel()
    
    # Choose a block size (must be a power of 2)
    BLOCK_SIZE = 1024
    
    # Grid size: number of blocks needed
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    add_kernel[grid](
        x_ptr=x,
        const_val=const_val,
        output_ptr=output,
        N=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


@pytest.mark.parametrize("vector_size,const_val,dtype", [
    (32, 3.5, torch.float32),           # Small vector
    (2048, -10.0, torch.float32),       # Vector divisible by block size
    (1234, 7.777, torch.float32),       # Vector not divisible by block size (tests mask)
    (5003, 1.5, torch.float32),         # Large vector not divisible by block size
    (256, 2.0, torch.float16),          # Float16
    (1023, 0.5, torch.float32),         # One less than block size (tests mask)
])
def test_const_add(vector_size, const_val, dtype):
    """Test Task 2: Constant Add with Block and Masking."""
    x = torch.randn(vector_size, dtype=dtype, device='cuda')
    
    torch_result = add_torch(x, const_val)
    triton_result = add_triton(x, const_val)
    
    assert torch.allclose(torch_result, triton_result, rtol=1e-4, atol=1e-5), \
        f"Results don't match: size={vector_size}, const={const_val}, dtype={dtype}"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Triton requires CUDA.")
    
    # Run tests with pytest from command line or run basic test
    pytest.main([__file__, "-v"])

