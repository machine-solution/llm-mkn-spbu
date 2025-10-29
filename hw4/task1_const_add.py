import pytest
import torch
import triton
import triton.language as tl


def add_torch(x, const_val):
    """Reference PyTorch implementation."""
    return x + const_val


@triton.jit
def add_kernel(x_ptr, const_val, output_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for adding a constant to a vector.
    
    Args:
        x_ptr: Pointer to input tensor
        const_val: Constant value to add
        output_ptr: Pointer to output tensor
        BLOCK_SIZE: Number of elements processed by each program instance
    """
    # Get the program ID for this instance
    pid = tl.program_id(0)
    
    # Compute the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Load data from global memory
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    
    # Add the constant
    output = x + const_val
    
    # Store the result back to global memory
    tl.store(output_ptr + offsets, output)


def add_triton(x, const_val):
    """
    Triton implementation of constant addition.
    
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
    # For this task, B0 == N0, so we use the smallest power of 2 >= n_elements
    # But actually, we can just use 1024 as a reasonable block size
    BLOCK_SIZE = 1024
    
    # Grid size: number of blocks needed
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    add_kernel[grid](
        x_ptr=x,
        const_val=const_val,
        output_ptr=output,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


@pytest.mark.parametrize("vector_size,const_val,dtype", [
    (32, 3.5, torch.float32),           # Small vector
    (2048, -10.0, torch.float32),       # Vector divisible by block size
    (1234, 7.777, torch.float32),       # Vector not divisible by block size
    (512, 1.5, torch.float32),          # Float32
    (256, 2.0, torch.float16),          # Float16
])
def test_const_add(vector_size, const_val, dtype):
    """Test Task 1: Constant Add with various configurations."""
    x = torch.randn(vector_size, dtype=dtype, device='cuda')
    
    torch_result = add_torch(x, const_val)
    triton_result = add_triton(x, const_val)
    
    assert torch.allclose(torch_result, triton_result, rtol=1e-4, atol=1e-5), \
        f"Results don't match for size={vector_size}, const={const_val}, dtype={dtype}"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Triton requires CUDA.")
    
    # Run tests with pytest from command line or run basic test
    pytest.main([__file__, "-v"])

