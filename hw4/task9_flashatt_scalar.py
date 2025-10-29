import pytest
import torch
import triton
import triton.language as tl


def flashatt_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
	"""Reference scalar single-head FlashAttention using stable softmax.
	q, k, v: (T,)
	Returns: (T,) where out[i] = sum_j softmax(q[i]*k[j]) * v[j]
	"""
	T = q.numel()
	x = q.view(T, 1) * k.view(1, T)
	x_max = x.max(1, keepdim=True).values
	x = x - x_max
	x_exp = torch.exp(x)
	soft = x_exp / x_exp.sum(1, keepdim=True)
	return (v.view(1, T) * soft).sum(1)


@triton.jit
def flashatt_scalar_kernel(
	q_ptr, k_ptr, v_ptr, out_ptr,
	T: tl.constexpr,
	STRIDE_Q: tl.constexpr,
	STRIDE_K: tl.constexpr,
	STRIDE_V: tl.constexpr,
	B1: tl.constexpr,
):
	"""Scalar FlashAttention with tiled streaming softmax.
	One program per query index i; loop over key/value tiles of size B1.
	Accumulates in fp32 using exp2 with log2(e)=1.4426950408889634.
	"""
	pid = tl.program_id(0)  # i
	if pid >= T:
		return

	# Load q[i]
	qi = tl.load(q_ptr + pid * STRIDE_Q).to(tl.float32)

	# Running vars
	m = tl.full((), -float('inf'), dtype=tl.float32)
	l = tl.zeros((), dtype=tl.float32)
	acc = tl.zeros((), dtype=tl.float32)

	cols = tl.arange(0, B1)

	for t_start in range(0, T, B1):
		j = t_start + cols
		mask = j < T
		k_tile = tl.load(k_ptr + j * STRIDE_K, mask=mask, other=0.0).to(tl.float32)
		v_tile = tl.load(v_ptr + j * STRIDE_V, mask=mask, other=0.0).to(tl.float32)

		# Scores for this tile
		s = qi * k_tile  # (B1,)
		# Update running max using tile max
		s_max = tl.max(tl.where(mask, s, -float('inf')), axis=0)
		m_new = tl.maximum(m, s_max)

		# Compute factors
		log2e = 1.4426950408889634
		a = tl.exp2((m - m_new) * log2e)
		p = tl.exp2((s - m_new) * log2e) * tl.where(mask, 1.0, 0.0)

		# Update running sum and accumulator
		l = l * a + tl.sum(p, axis=0)
		acc = acc * a + tl.sum(v_tile * p, axis=0)

		m = m_new

	# Write result
	out = acc / l
	tl.store(out_ptr + pid, out)


def flashatt_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
	"""Triton implementation wrapper for scalar FlashAttention."""
	q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
	assert q.dim() == k.dim() == v.dim() == 1
	T = q.numel()
	out = torch.empty((T,), dtype=torch.float32, device=q.device)

	B1 = 256
	grid = (triton.cdiv(T, 1),)

	flashatt_scalar_kernel[grid](
		q_ptr=q,
		k_ptr=k,
		v_ptr=v,
		out_ptr=out,
		T=T,
		STRIDE_Q=q.stride(0),
		STRIDE_K=k.stride(0),
		STRIDE_V=v.stride(0),
		B1=B1,
	)
	return out.to(q.dtype)


@pytest.mark.parametrize("T", [1, 17, 64, 257, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_flashatt_scalar(T, dtype):
	q = torch.randn(T, dtype=dtype, device='cuda')
	k = torch.randn(T, dtype=dtype, device='cuda')
	v = torch.randn(T, dtype=dtype, device='cuda')

	ref = flashatt_torch(q.float(), k.float(), v.float()).to(dtype)
	tri = flashatt_triton(q, k, v)

	assert torch.allclose(ref, tri, rtol=1e-4 if dtype==torch.float32 else 2e-3, atol=1e-5 if dtype==torch.float32 else 2e-3)


if __name__ == "__main__":
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA is not available. Triton requires CUDA.")
	pytest.main([__file__, "-v"])
