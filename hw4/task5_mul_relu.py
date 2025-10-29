import pytest
import torch
import triton
import triton.language as tl


def mul_relu_block_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	"""Reference PyTorch implementation: outer product + ReLU.
	Returns: (N1, N0) = relu(y[:, None] * x[None, :])
	"""
	return torch.relu(x[None, :] * y[:, None])


@triton.jit
def mul_relu_kernel(
	x_ptr,
	y_ptr,
	out_ptr,
	N0: tl.constexpr,
	N1: tl.constexpr,
	BLOCK_SIZE_ROW: tl.constexpr,
	BLOCK_SIZE_COL: tl.constexpr,
):
	"""Fused outer multiplication + ReLU with 2D blocking and masks.
	- Program grid is 2D: (rows, cols)
	- B0 < N0, B1 < N1: masks on both axes
	"""
	row_pid = tl.program_id(0)
	col_pid = tl.program_id(1)

	row_start = row_pid * BLOCK_SIZE_ROW
	col_start = col_pid * BLOCK_SIZE_COL

	row_idx = row_start + tl.arange(0, BLOCK_SIZE_ROW)
	col_idx = col_start + tl.arange(0, BLOCK_SIZE_COL)

	row_mask = row_idx < N1
	col_mask = col_idx < N0

	# Load tiles from vectors with masks
	x_vals = tl.load(x_ptr + col_idx, mask=col_mask, other=0.0)  # shape: (BLOCK_SIZE_COL,)
	y_vals = tl.load(y_ptr + row_idx, mask=row_mask, other=0.0)  # shape: (BLOCK_SIZE_ROW,)

	# Broadcast to 2D tile
	x_2d = x_vals[None, :]  # (1, BLOCK_SIZE_COL)
	y_2d = y_vals[:, None]  # (BLOCK_SIZE_ROW, 1)

	# Fused op: outer product then ReLU
	prod = x_2d * y_2d
	zero = tl.zeros_like(prod)
	relu = tl.maximum(prod, zero)

	# Compute flattened output offsets for masked store
	row_idx_2d = row_idx[:, None]
	col_idx_2d = col_idx[None, :]
	out_offsets = row_idx_2d * N0 + col_idx_2d

	out_mask = row_mask[:, None] & col_mask[None, :]
	tl.store(out_ptr + out_offsets, relu, mask=out_mask)


def mul_relu_block_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	"""Triton implementation wrapper."""
	x = x.contiguous()
	y = y.contiguous()
	assert x.dim() == 1 and y.dim() == 1
	N0 = x.numel()
	N1 = y.numel()

	out = torch.empty((N1, N0), dtype=x.dtype, device=x.device)

	BLOCK_SIZE_ROW = 32
	BLOCK_SIZE_COL = 32
	grid = (
		(N1 + BLOCK_SIZE_ROW - 1) // BLOCK_SIZE_ROW,
		(N0 + BLOCK_SIZE_COL - 1) // BLOCK_SIZE_COL,
	)

	mul_relu_kernel[grid](
		x_ptr=x,
		y_ptr=y,
		out_ptr=out,
		N0=N0,
		N1=N1,
		BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
		BLOCK_SIZE_COL=BLOCK_SIZE_COL,
	)
	return out


@pytest.mark.parametrize("n0,n1", [
	(32, 16),
	(64, 32),
	(128, 128),
	(256, 512),
	(1024, 512),
	(33, 65),     # not divisible
	(1, 1),       # edge
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mul_relu_block(n0, n1, dtype):
	x = torch.randn(n0, dtype=dtype, device='cuda')
	y = torch.randn(n1, dtype=dtype, device='cuda')

	torch_out = mul_relu_block_torch(x, y)
	triton_out = mul_relu_block_triton(x, y)

	assert torch.allclose(torch_out, triton_out, rtol=1e-4, atol=1e-5), \
		f"Mismatch for n0={n0}, n1={n1}, dtype={dtype}"


if __name__ == "__main__":
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is not available. Triton requires CUDA.")
	pytest.main([__file__, "-v"])
