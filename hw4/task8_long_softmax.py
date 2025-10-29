import pytest
import torch
import triton
import triton.language as tl


def softmax_torch(x: torch.Tensor) -> torch.Tensor:
	"""Reference PyTorch implementation: row-wise softmax (stable)."""
	x_max = x.max(1, keepdim=True).values
	x_centered = x - x_max
	x_exp = torch.exp(x_centered)
	return x_exp / x_exp.sum(1, keepdim=True)


@triton.jit
def row_softmax_kernel(
	x_ptr,          # (N0, T)
	out_ptr,        # (N0, T)
	N0: tl.constexpr,
	T: tl.constexpr,
	STRIDE_N0: tl.constexpr,
	STRIDE_T: tl.constexpr,
	B1: tl.constexpr,   # tile size along T
):
	"""Row-wise softmax with three tiled passes per row for stability.
	Pass 1: row_max
	Pass 2: row_sum = sum(exp2((x - row_max) * log2e))
	Pass 3: write exp2((x - row_max) * log2e) / row_sum
	"""
	pid = tl.program_id(0)  # row index i
	if pid >= N0:
		return

	row_base = pid * STRIDE_N0
	col_offsets = tl.arange(0, B1)

	# Pass 1: compute row max
	row_max = tl.full((), -float('inf'), dtype=tl.float32)
	for t_start in range(0, T, B1):
		cols = t_start + col_offsets
		mask = cols < T
		ptrs = x_ptr + row_base + cols * STRIDE_T
		vals = tl.load(ptrs, mask=mask, other=-float('inf'))
		row_max = tl.maximum(row_max, tl.max(vals.to(tl.float32), axis=0))

	# Pass 2: compute denominator (sum of exponentials)
	row_sum = tl.zeros((), dtype=tl.float32)
	for t_start in range(0, T, B1):
		cols = t_start + col_offsets
		mask = cols < T
		ptrs = x_ptr + row_base + cols * STRIDE_T
		vals = tl.load(ptrs, mask=mask, other=-float('inf')).to(tl.float32)
		centered = vals - row_max
		# exp(x) = 2^{x * log2(e)}; use inline literal for log2(e)
		exp2_vals = tl.exp2(centered * 1.4426950408889634)
		row_sum += tl.sum(exp2_vals, axis=0)

	# Pass 3: write normalized outputs
	row_sum_inv = 1.0 / row_sum
	for t_start in range(0, T, B1):
		cols = t_start + col_offsets
		mask = cols < T
		ptrs = x_ptr + row_base + cols * STRIDE_T
		vals = tl.load(ptrs, mask=mask, other=-float('inf')).to(tl.float32)
		centered = vals - row_max
		exp2_vals = tl.exp2(centered * 1.4426950408889634)
		softmax_tile = exp2_vals * row_sum_inv
		out_ptrs = out_ptr + row_base + cols * STRIDE_T
		tl.store(out_ptrs, softmax_tile.to(tl.float32), mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
	"""Triton implementation wrapper: row-wise softmax with tiling."""
	x = x.contiguous()
	assert x.dim() == 2
	N0, T = x.shape
	# Use float32 output for numerical accuracy, then cast to dtype of x
	out = torch.empty((N0, T), dtype=torch.float32, device=x.device)

	B1 = 256
	stride_n0 = x.stride(0)
	stride_t = x.stride(1)
	grid = (triton.cdiv(N0, 1),)

	row_softmax_kernel[grid](
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
	(8, 257),
	(32, 1024),
	(64, 1537),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_row_softmax(N0, T, dtype):
	x = torch.randn(N0, T, dtype=dtype, device='cuda')
	t_ref = softmax_torch(x.float())
	t_tri = softmax_triton(x)
	assert torch.allclose(t_ref.to(dtype), t_tri, rtol=1e-4, atol=1e-6 if dtype==torch.float32 else 1e-4)


if __name__ == "__main__":
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is not available. Triton requires CUDA.")
	pytest.main([__file__, "-v"])
