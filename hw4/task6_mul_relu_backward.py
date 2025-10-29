import pytest
import torch
import triton
import triton.language as tl


def mul_relu_block_back_torch(x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
	"""Reference PyTorch backward: dx for z = relu(x * y[:, None]).
	x: (N0,), y: (N1,), dz: (N1, N0)
	Returns dx: (N0,)
	"""
	x = x.clone().detach().requires_grad_(True)
	y = y.clone().detach().requires_grad_(True)
	z = torch.relu(x[None, :] * y[:, None])  # (N1, N0)
	z.backward(dz)
	dx = x.grad
	return dx


@triton.jit
def mul_relu_back_kernel(
	x_ptr,    # (N0,)
	y_ptr,    # (N1,)
	dz_ptr,   # (N1, N0)
	dx_ptr,   # (N0,) float32 buffer
	N0: tl.constexpr,
	N1: tl.constexpr,
	BLOCK_SIZE_ROW: tl.constexpr,
	BLOCK_SIZE_COL: tl.constexpr,
):
	"""Compute dx = sum_j 1[u>0] * y[j] * dz[j, i], where u = x[i] * y[j].
	Accumulates in float32 for numerical stability.
	"""
	row_pid = tl.program_id(0)
	col_pid = tl.program_id(1)

	row_start = row_pid * BLOCK_SIZE_ROW
	col_start = col_pid * BLOCK_SIZE_COL

	row_idx = row_start + tl.arange(0, BLOCK_SIZE_ROW)
	col_idx = col_start + tl.arange(0, BLOCK_SIZE_COL)

	row_mask = row_idx < N1
	col_mask = col_idx < N0

	# Load vector slices and upcast to fp32
	x_vals = tl.load(x_ptr + col_idx, mask=col_mask, other=0.0)
	y_vals = tl.load(y_ptr + row_idx, mask=row_mask, other=0.0)
	x_vals_f32 = x_vals.to(tl.float32)
	y_vals_f32 = y_vals.to(tl.float32)

	# Load dz tile and upcast to fp32
	row_idx_2d = row_idx[:, None]
	col_idx_2d = col_idx[None, :]
	dz_offsets = row_idx_2d * N0 + col_idx_2d
	dz_tile = tl.load(dz_ptr + dz_offsets, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
	dz_tile_f32 = dz_tile.to(tl.float32)

	# Compute u = x[i] * y[j] in fp32
	x_2d = x_vals_f32[None, :]
	y_2d = y_vals_f32[:, None]
	u = x_2d * y_2d

	# ReLU derivative mask
	relu_mask = u > 0

	# Contribution in fp32
	contrib = y_2d * dz_tile_f32
	contrib = tl.where(relu_mask, contrib, 0.0)

	# Reduce over rows to partial dx (fp32)
	partial_dx = tl.sum(contrib, axis=0)

	# Atomic add into float32 dx buffer
	tl.atomic_add(dx_ptr + col_idx, partial_dx, mask=col_mask)


def mul_relu_block_back_triton(x: torch.Tensor, y: torch.Tensor, dz: torch.Tensor) -> torch.Tensor:
	"""Triton wrapper for backward dx of fused outer mul + ReLU.
	Inputs:
	- x: (N0,)
	- y: (N1,)
	- dz: (N1, N0)
	Returns:
	- dx: (N0,)
	"""
	x = x.contiguous()
	y = y.contiguous()
	dz = dz.contiguous()
	assert x.dim() == 1 and y.dim() == 1 and dz.dim() == 2
	N0 = x.numel()
	N1 = y.numel()
	assert dz.shape == (N1, N0)

	# Accumulate in float32 for stability
	dx_f32 = torch.zeros((N0,), dtype=torch.float32, device=x.device)

	BLOCK_SIZE_ROW = 64
	BLOCK_SIZE_COL = 64
	grid = (
		(N1 + BLOCK_SIZE_ROW - 1) // BLOCK_SIZE_ROW,
		(N0 + BLOCK_SIZE_COL - 1) // BLOCK_SIZE_COL,
	)

	mul_relu_back_kernel[grid](
		x_ptr=x,
		y_ptr=y,
		dz_ptr=dz,
		dx_ptr=dx_f32,
		N0=N0,
		N1=N1,
		BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
		BLOCK_SIZE_COL=BLOCK_SIZE_COL,
	)
	return dx_f32.to(x.dtype)


@pytest.mark.parametrize("n0,n1", [
	(32, 16),
	(64, 32),
	(128, 128),
	(257, 129),
	(1000, 513),
])
@pytest.mark.parametrize("dtype,rtol,atol", [
	(torch.float32, 1e-4, 1e-5),
	(torch.float16, 2e-3, 2e-3),
])
def test_mul_relu_backward(n0, n1, dtype, rtol, atol):
	x = torch.randn(n0, dtype=dtype, device='cuda')
	y = torch.randn(n1, dtype=dtype, device='cuda')
	dz = torch.randn(n1, n0, dtype=dtype, device='cuda')

	dx_torch = mul_relu_block_back_torch(x.float(), y.float(), dz.float()).to(dtype)
	dx_triton = mul_relu_block_back_triton(x, y, dz)

	assert torch.allclose(dx_torch, dx_triton, rtol=rtol, atol=atol), \
		f"Mismatch for n0={n0}, n1={n1}, dtype={dtype}: max abs diff={(dx_torch-dx_triton).abs().max().item()}"


if __name__ == "__main__":
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is not available. Triton requires CUDA.")
	pytest.main([__file__, "-v"])
