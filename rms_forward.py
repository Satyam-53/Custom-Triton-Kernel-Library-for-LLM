import torch
import triton
import triton.language as tl

# @triton.jit
# def rms_forward_kernel():

'''
RMSNorm Triton Kernel — Optimizations

  3 issues with naive kernel:

  1. No streaming — BLOCK_N = N means huge register pressure for large d_model (e.g. 8192)
  2. One-pass is impossible — rstd depends on full row sum, so you must see all x before normalizing
  3. Gamma reloaded every row — same data loaded by every program instance

  Fixes:

  1. Stream with smaller BLOCK_N — loop over tiles, accumulate sum(x²) across tiles
  2. Two loops, one kernel — rstd is a scalar register between loops, no HBM round-trip
  3. Gamma: rely on L2 cache — gamma is small + read-only, hot in L2 after first few rows; can't fully avoid the instruction

  Key insight:

  ▎ Two passes ≠ two kernel launches. rstd lives in a register between the two tile loops.

  Pass 1: stream tiles → accumulate ss → compute rstd (scalar, register)
  Pass 2: stream tiles → load x, gamma → x * rstd * γ → store

  Memory hierarchy reminder:
  - HBM → slow, avoid redundant round-trips
  - L2 cache → small read-only tensors (like gamma) stay hot here
  - Registers → cheapest, use for scalars like rstd
'''

@triton.jit
def rmsnorm_kernel_v2(
    x_ptr, gamma_ptr, out_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    row_off = row * N

    # Pass 1: stream over tiles, accumulate sum of squares
    ss = tl.zeros([1], dtype=tl.float32)
    for col_start in tl.range(0, N, BLOCK_N):
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        ss += tl.sum(x_fp32 * x_fp32, axis=0)

    rstd = tl.rsqrt(ss / N + eps)  # scalar, lives in register

    # Pass 2: stream again, normalize + apply gamma
    for col_start in tl.range(0, N, BLOCK_N):
        cols = col_start + tl.arange(0, BLOCK_N)
    rstd = tl.rsqrt(ss / N + eps)  # scalar, lives in register

    # Pass 2: stream again, normalize + apply gamma
    for col_start in tl.range(0, N, BLOCK_N):
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0)
        gamma = tl.load(gamma_ptr + cols, mask=mask, other=1.0)  # L2 cached
        out = x.to(tl.float32) * rstd * gamma
        tl.store(out_ptr + row_off + cols, out.to(x.dtype), mask=mask)