1.  tl.constexpr is for the things that need to be known at the compile time so that struture could be organized like BLOCK_SIZE.

2.  tl.program_id(0) is the id of the program, which is the index of the block.

3. What is numerically tricky about naive sigmoid?
    naive sigmoid can overflow for very negative x.
    so there is a trick used for it. [IMPORTANT FOR INTERVIEW]
    
4. Triton's grid must be a 1-, 2-, or 3-tuple corresponding to the number of launch dimensions. [IMPORTANT]

5. How meta works

  When you launch a kernel like:
  affine_transform_kernel[grid](x, a, b, out, n, BLOCK_SIZE=block_size)

  Triton's JIT sees that BLOCK_SIZE is annotated as tl.constexpr in the kernel signature. Before dispatching the launch,
  it collects all constexpr kwargs into a dict and passes that dict as meta to your grid callable. So
  meta["BLOCK_SIZE"] equals whatever you passed as BLOCK_SIZE=block_size.

  You never build that dict explicitly — Triton does it automatically from the tl.constexpr-annotated parameters. The
  reason the grid lambda needs BLOCK_SIZE is that the number of program instances depends on the tile size, and tile
  size is a compile-time constant, so both must be resolved together before launch.

6. There is a difference between how you use tl.rand/randn and torch.rand/randn. torch methods support seed set somewhere else, but for triton.language you need to send it explicitly in every function call and it makes sense why it was designed this way. 

7.  Dropout mask generation — two approaches

  ---
  1. Pre-generate in wrapper (torch.bernoulli)
  d_mask = torch.bernoulli(torch.full_like(x, 1.0 - p))  # float mask of 0.0/1.0
  - Mask generated on CPU/GPU before kernel launch, passed as a pointer arg
  - Kernel just loads it like any other tensor
  - Must store mask for backward → O(n) activation memory per layer

  ---
  2. Generate inside kernel (tl.rand)
  rand = tl.rand(seed, offsets)   # seed: int, offsets: per-element unique key
  d_mask = (rand > p).to(tl.float32)
  - seed passed as a scalar kernel arg, saved in ctx during forward
  - Backward regenerates identical mask by reusing same seed + offsets
  - Only O(1) memory (just the seed integer)

  ---
  Trade-off summary

  ┌──────────────┬──────────────────┬────────────────────────────────────────────────┐
  │              │ torch.bernoulli  │                    tl.rand                     │
  ├──────────────┼──────────────────┼────────────────────────────────────────────────┤
  │ Mask storage │ O(n)             │ O(1) — seed only                               │
  ├──────────────┼──────────────────┼────────────────────────────────────────────────┤
  │ Complexity   │ Simple           │ Need RNG management                            │
  ├──────────────┼──────────────────┼────────────────────────────────────────────────┤
  │ Backward     │ Load stored mask │ Regenerate from seed                           │
  ├──────────────┼──────────────────┼────────────────────────────────────────────────┤
  │ Used in      │ Simple kernels   │ Memory-efficient kernels (e.g. FlashAttention) │
  └──────────────┴──────────────────┴────────────────────────────────────────────────┘

  The offsets array in tl.rand acts as a per-element unique key — same (seed, offset) pair always produces the same random
  value, which is what makes backward regeneration exact.


8. gelu out using the tanh approximation- 
  out = 0.5 * x ( 1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  - This is a very common activation function in deep learning models. It's used in transformers, GPT, etc.

9. The primary motivation for introducing GELU (Gaussian Error Linear Unit) was to merge the concepts of neural network dropouts, zoneouts, and activation functions into a single probabilistic framework. However, it naturally solves the "dead ReLU" issue as a beneficial side effect.

10.  tl.sqrt(2 / pi) — calling a Triton math op on a Python scalar

  tl.sqrt is meant for Triton tensors. Calling it on a Python float is fragile and hits every launch. Precompute it once in
  Python:

  SQRT_2_OVER_PI = 0.7978845608028654  # computed once, used as a scalar constant
  inner = SQRT_2_OVER_PI * (x_fp32 + 0.044715 * x_fp32 * x_fp32 * x_fp32)

11. Stable Sigmoid

  Naive 1 / (1 + exp(-x)) overflows when x << 0 because exp(-x) → inf.

  Fix: branch on sign so the exp argument is always negative (bounded):

  x >= 0:  1 / (1 + exp(-x))      # -x <= 0, safe
  x <  0:  exp(x) / (1 + exp(x))  # x < 0, safe

  Same mathematical value, but exp is only ever called with a non-positive argument, so it never exceeds 1. No overflow
  possible.

  In Triton, use tl.where to avoid a branch:

  x_fp32 = x.to(tl.float32)
  safe_exp = tl.exp(-tl.abs(x_fp32))
  out = tl.where(x_fp32 >= 0, 1.0 / (1.0 + safe_exp), safe_exp / (1.0 + safe_exp))

12.  torch.empty_like(x) defaults to memory_format=torch.preserve_format, so it copies the strides from     x.     If  x is non-contiguous (e.g. a transposed tensor), the output buffer is also non-contiguous.

  This directly bites your Triton kernels. Your offset arithmetic:

  offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  tl.load(x_ptr + offsets, ...)
  tl.store(out_ptr + offsets, ...)

  assumes elements are laid out contiguously in memory — stride 1. If x is non-contiguous, you're reading/writing wrong
  memory locations. No error, just silently wrong results.

13. to get strides, you simply do tensor.stride(0), tensor.stride(1) and so on....

14.  **General:** offset = Σᵢ (indexᵢ · strideᵢ)

  **2D:** offset = i·s₀ + j·s₁

  **4D:** offset = b·sᵦ + h·sₕ + m·sₘ + n·sₙ

15. Simple rule:

  ┌──────────────────────────────────────────┬────────────────────────────────────┐
  │                 Used for                 │          Needs constexpr?          │
  ├──────────────────────────────────────────┼────────────────────────────────────┤
  │ tl.arange, tensor shapes, loop bounds    │ Yes                                │
  ├──────────────────────────────────────────┼────────────────────────────────────┤
  │ Arithmetic, comparisons, pointer offsets │ No                                 │
  ├──────────────────────────────────────────┼────────────────────────────────────┤
  │ Launch grid size (set in Python)         │ Never — it's not even a kernel arg │
  └──────────────────────────────────────────┴────────────────────────────────────┘

  So BLOCK_N is constexpr because it defines a tensor shape inside the kernel. M and N are just runtime values.

16. tiled stable streaming fashion softmax-  Key interview talking points:

  - Why 2 passes? You need the global (m, l) statistics before you can normalize any element. Flash Attention avoids this by
  fusing the softmax into the output accumulation O += softmax * V.
  - Why alpha = exp(m_old - m_new)? It rescales the old running sum from being relative to m_old to relative to m_new.
  Without it, adding partial sums computed under different maxes is meaningless.
  - Why other=float('-inf') on masked loads? So tl.max and tl.sum over padding are identity-neutral — exp(-inf) = 0
  contributes nothing to the sum, and -inf loses to any real value in max.
  - BLOCK_N must be power of 2 — Triton requires constexpr block sizes to be powers of 2 for tl.arange.
  - The invariant the loop maintains: after each tile, l_old = sum_{all seen j} exp(x_j - m_old). The alpha correction is
  what preserves this invariant across tile boundaries.

17.  The real optimization lever isn't reducing passes — it's fusing the softmax backward with the surrounding matmuls, which is what Flash Attention does. That's where the actual memory bandwidth savings come from.

18.  Attention backward without fusion — the problem

  In attention, the forward is:

  S = Q @ K^T        (B, N, N)   ← this matrix is the problem
  P = softmax(S)     (B, N, N)
  O = P @ V          (B, N, d)

  To do backward unfused, you'd need to:

  1. load P      (B, N, N) from HBM   → compute dP = dO @ V^T
  2. load P      (B, N, N) from HBM   → compute dS = softmax_bwd(P, dP)
  3. load dS     (B, N, N) from HBM   → compute dQ = dS @ K
  4. load dS     (B, N, N) from HBM   → compute dK = dS^T @ Q
  5. load P      (B, N, N) from HBM   → compute dV = P^T @ dO

  For N=4096, P is 4096 x 4096 x 4 bytes = 64MB per head. You're reading it 3 times. This is pure memory bandwidth waste —
  HBM reads dominate runtime, not compute.

  ---
  What fusion does

  Flash Attention backward fuses steps 1-5 into a single kernel. For each tile of rows it processes, it:

  1. Load a tile of Q, K, V, dO from HBM — once
  2. Recompute S and P on-chip (SRAM) — free, avoids storing P
  3. Compute dP, dS, dQ, dK, dV all in registers
  4. Write dQ, dK, dV to HBM — once

  The (N, N) matrix P and S never land in HBM. They live and die in SRAM within a tile.

19.  RMSNorm — γ parameter

  - Shape: (d_model,) — one scalar per feature dimension
  - Same across tokens, different across features
  - Broadcast over all sequence positions — token-independent
  - Purpose: re-scale each feature after normalization (learned per feature, not per token)
  - No bias term (unlike LayerNorm), no mean subtraction — just rescale by RMS

  output = (x / RMS(x)) * γ    # γ shape: (d_model,)

  Intuition: Normalization strips magnitude info from each token. γ lets the model relearn "how important is feature i" —
  independently per feature. No per-token γ needed because token-specific behavior is handled by the surrounding weight
  matrices (attention, FFN), not the norm.

20. RMSNorm Triton Kernel — Optimizations

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

21. Whenever writin layer norm kernel, make sure you are using tl.where when centering the data with the mean.
  This is important since you might have 0.0s from the padding and you want to avoid them.
  Very important for the correctness of the layernorm kernel.

22. eps is usually 1e-5.

23. okay I think I implemented it. what's in the next day, if nothing else is left for the current day.

24. The core idea of causal masking is that at position i, the  model can only attend to positions 0 to i — never the future. This is what makes autoregressive generation possible in transformers. torch.triu with diagonal=1 generates the boolean mask of future positions, masked_fill with -float("inf") blocks those positions by zeroing them out after softmax, and [None, None, :, :] ensures the 2D mask correctly broadcasts over the full 4D scores tensor.

  bash```
  causal_mask = torch.triu(torch.ones((M, N), device=q.device, dtype=torch.bool), diagonal=1)

  scores = scores.masked_fill(causal_mask, -float("inf"))
  ```

25. For batched Multi head attn, we use 3d grid in Triton and squash the B, H dimension into B*H, then each can be taken out with the help of pid_bh and using the division and reminder trick.

26. GeLU: Exact vs Approximate — Quick Notes

  Exact:
  GeLU(x) = x · Φ(x) = x · 0.5·(1 + erf(x/√2))

  Approximate:
  GeLU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))

  Why approximate exists:
  - erf has no hardware instruction — computed via library, slow on GPU
  - tanh maps to a hardware-accelerated GPU op
  - Error is negligible (~0.0004 max), networks can't tell the difference

  Why it persists today (even though GPUs got faster):
  - GPT-2, BERT etc. were trained with the approximate version
  - Must match it exactly to load their checkpoints — compatibility lock-in

  PyTorch:
  F.gelu(x)                      # exact
  F.gelu(x, approximate='tanh')  # approximate

27.  Q: Why does Flash Attention use an accumulator for online softmax, but standalone stable softmax requires a second
  pass?

  Flash Attention — key assumptions:
  - Head dim d is small and fixed (typically 64–128) — this is what keeps tile sizes bounded in SRAM
  - You never load full Q, K, V — you load tiles: [Br, d] for Q, [Bc, d] for K and V, tiling over sequence length N
  - SRAM budget must fit 3 × Bc × d + Br × d simultaneously — Br, Bc are tuned to this
  - If d were large (256+), even one tile would spill to global memory and the whole scheme breaks

  Why the accumulator works in FA:
  - Computes softmax(QKᵀ)V fused — accumulator stores output o of shape [Br, d] (fixed, small)
  - When a new tile raises the running max m: rescale o_old *= exp(m_old - m_new), add new contribution — cost is O(Br ×
  d), cheap
  - Works because the dot product with V contracts the sequence dimension out of o — accumulator size is independent of
  N

  Why standalone softmax needs a second pass:
  - Output is the full probability vector — shape [seq_len], grows with N
  - When a later block raises the max, every previously written value is stale by exp(m_old - m_new)
  - Fixing them requires a full re-read of all prior outputs — that is the second pass
  - No way around it without holding all intermediate exp values in SRAM, which defeats streaming

  Core insight:
  FA's accumulator stays small because V multiplication collapses the sequence dimension to d. Standalone softmax has no
  such contraction — online correction costs O(seq_len) per block update, equivalent to a full second pass.

28. Last Minute Interview Notes
1. Trailing Comma in Single Element Tuples
(M) is just an integer, (M,) is a tuple. All PyTorch tensor creation functions expect a tuple for shape, so trailing comma is required for 1D tensors. For 2D and above the comma between dimensions already makes it a tuple:


Apply
torch.zeros((M,))   # correct
torch.zeros((M))    # TypeError
torch.zeros((M, N)) # correct, no trailing comma needed
2. torch.zeros vs torch.empty
Use torch.zeros when initial value matters, use torch.empty when you are going to overwrite all values anyway (e.g. output buffer for a kernel). torch.empty is slightly faster since it skips initialization:


Apply
acc = torch.zeros((M, N), dtype=torch.float32)   # initial value matters
output = torch.empty((M, N), dtype=torch.float32) # will be overwritten
3. torch.max vs torch.maximum
torch.max with dim reduces along a dimension and returns a namedtuple — always use .values to extract the tensor. torch.maximum is elementwise between two tensors:


Apply
m_block = torch.max(scores, dim=-1, keepdim=True).values  # (B, H, M, 1)
m_new = torch.maximum(m_old, m_block)                     # elementwise
4. None Indexing
Inserts a new dimension of size 1, zero cost view, no data copied. Used to make shapes compatible for broadcasting. Never valid inside a shape tuple:


Apply
mask[None, None, :, :]  # (M, N) -> (1, 1, M, N), broadcasts over (B, H, M, N)
torch.zeros((M, None))  # TypeError
5. Causal Masking
torch.triu with diagonal=1 marks future positions as True. masked_fill replaces True positions with -inf so softmax turns them to 0:


Apply
causal_mask = torch.triu(torch.ones((M, N), device=q.device, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(causal_mask[None, None, :, :], -float("inf"))
attn_weights = F.softmax(scores, dim=-1)  # -inf becomes 0
6. Dtypes Inside Triton Kernels
No device argument needed inside kernels — already on GPU. Hardcode tl.float32 for accumulators, pass input dtype as tl.constexpr. tl.full requires dtype unlike PyTorch:


Apply
@triton.jit
def kernel(..., DTYPE: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) # always float32
    a = tl.load(a_ptr).to(DTYPE)                         # input dtype
    tl.store(output_ptr, acc.to(DTYPE))                  # cast back

tl.full((BLOCK_M,), -float("inf"), tl.float32)  # dtype required in tl.full

29. Question: "What happens when you launch a Triton kernel?"

  ---
  The key thing to understand is that @triton.jit doesn't compile anything — it just wraps your function into a
  JITFunction object at import time.

  The actual process is two-stage lazy execution:

  Stage 1 — kernel[grid]
  This is just Python's __getitem__. It captures the grid callable and returns a launcher object. Nothing runs on the
  GPU yet.

  Stage 2 — launcher(...)
  This is where the real work happens. Triton first separates your arguments into runtime values (tensors, scalars) and
  compile-time constants (tl.constexpr). It builds a cache key from those constexpr values plus dtype and device. On a
  cache miss it compiles to PTX/CUBIN; on a hit it skips straight to launch. Then it builds the meta dict from the
  resolved constexpr values and calls your grid lambda with it — that's the moment grid(meta) executes. Finally it
  dispatches the compiled kernel to the GPU.

  Why does the grid need to be a lambda at all?

  Because the grid size often depends on block size (cdiv(M, BLOCK_M)), but block size is a constexpr that's only
  resolved at the call site. The lambda defers grid computation until Triton has those constants. If your grid doesn't
  depend on any constexpr, you can just pass a plain tuple and skip the lambda entirely.

  The one-liner answer if they push you:

  ▎ "Triton JIT-compiles per unique constexpr specialization, caches the result, then calls your grid lambda with the
  ▎ resolved constants right before GPU dispatch."

30.  Grid design (Triton)
  - One program per (batch, seq, head) — fully independent
  - D dimension is small (32–128), fits in registers in one shot — no tiling needed
  - Tile along S for long-context prefill, tune BLOCK_S (16–32 typical)
  - Decoding: S=1 naturally, so grid is just (B × H,)

  float32 precision
  - inv_freq starts as float32 but model.to(bf16) casts all buffers including it
  - bf16 has only 7 mantissa bits — precision loss in cos/sin corrupts positional encoding at long contexts
  - Fix: .float() before computing freqs, or upcast after torch.cos()

  Things that trip people up
  - tl.arange(0, N) requires N to be power of 2 in Triton
  - Pass strides explicitly — Triton won't handle non-contiguous tensors automatically
  - cos/sin table shape: [seq_len, head_dim/2] — shared across heads, no head stride needed

31. “I cannot generally reuse Q’s cos/sin for K by transposing because RoPE depends on token position. Q positions and K positions are not necessarily the same, especially when a fixed Q block loops over many K blocks. I load cos/sin for Q using query positions and cos/sin for K using key positions. If K is loaded transposed, I load K’s cos/sin in the same transposed [D/2, BLOCK_N] shape.”

32. 



AT THE END MAKE SURE YOU KNOW HOW TO WRITE TEST AND AUTOTUNE METHODS, USING TOLERANCES AS WELL AND TORCH REFERENCE FUNCTIONS.
