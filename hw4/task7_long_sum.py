import pytest
import torch
import triton
import triton.language as tl


def sum_torch(x: torch.Tensor) -> torch.Tensor:
	"""Reference PyTorch implementation: row-wise sum.
	Input: x of shape (N0, T)
	Returns: (N0,)
	"""
	return x.sum(1)


@triton.jit
def row_sum_kernel(
	x_ptr,          # (N0, T)
	out_ptr,        # (N0,)
	N0: tl.constexpr,
	T: tl.constexpr,
	STRIDE_N0: tl.constexpr,
	STRIDE_T: tl.constexpr,
	B1: tl.constexpr,   # tile size along T
):
	"""Sum each row with inner tiling over T.
	Grid: 1D over rows (pid = i in [0, N0)).
	For each row, loop over tiles of length B1 along T with masking at tail.
	"""
	pid = tl.program_id(0)  # row index i
	# Guard for pid in case grid > N0
	if pid >= N0:
		return

	# Running accumulator in fp32 for stability
	acc = tl.zeros((), dtype=tl.float32)

	# Column offsets within a tile [0..B1)
	col_offsets = tl.arange(0, B1)

	# Base pointer for this row start (row-major layout): i * STRIDE_N0
	row_base = pid * STRIDE_N0

	# Loop over tiles along T
	for t_start in range(0, T, B1):
		cols = t_start + col_offsets  # absolute column indices
		mask = cols < T
		# Compute element pointers for this tile
		ptrs = x_ptr + row_base + cols * STRIDE_T
		vals = tl.load(ptrs, mask=mask, other=0.0)
		# Sum this tile (promote to fp32)
		tile_sum = tl.sum(vals.to(tl.float32), axis=0)
		acc += tile_sum

	# Store result (cast back to input dtype width conveniently via out dtype)
	tl.store(out_ptr + pid, acc)


def sum_triton(x: torch.Tensor) -> torch.Tensor:
	"""Triton implementation wrapper: row-wise sum with tiling."""
	x = x.contiguous()
	assert x.dim() == 2
	N0, T = x.shape
	out = torch.empty((N0,), dtype=torch.float32, device=x.device)

	# Choose tile size along T
	B1 = 256

	# Strides in elements (row-major contiguous):
	# For a 2D contiguous tensor of shape (N0, T), stride(0) == T, stride(1) == 1
	stride_n0 = x.stride(0)
	stride_t = x.stride(1)

	# 1D grid over rows
	grid = (triton.cdiv(N0, 1),)

	row_sum_kernel[grid](
		x_ptr=x,
		out_ptr=out,
		N0=N0,
		T=T,
		STRIDE_N0=stride_n0,
		STRIDE_T=stride_t,
		B1=B1,
	)
	return out.to(x.dtype)


@pytest.mark.parametrize("N0,T", [
	(1, 1),
	(4, 32),
	(8, 257),     # not divisible by tile
	(32, 1024),
	(64, 1537),   # not divisible by tile
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_row_sum(N0, T, dtype):
	x = torch.randn(N0, T, dtype=dtype, device='cuda')
	t_ref = sum_torch(x.float())
	t_tri = sum_triton(x)
	assert torch.allclose(t_ref.to(dtype), t_tri, rtol=1e-4 if dtype==torch.float32 else 2e-3, atol=1e-5 if dtype==torch.float32 else 2e-3)


if __name__ == "__main__":
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is not available. Triton requires CUDA.")
	pytest.main([__file__, "-v"])
