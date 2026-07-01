'''
We'll first implement the single head triton flash attention 
forward pass in a way that's very similar to the torch version
of the attention function, but using triton instead of pytorch.
'''

import torch
import triton
import triton.language as tl


'''
Q: M X D
K: N X D
V: N X D

If causal attn, assert M == N, can/should be handled by the wrapper method. Kernel
can be agnostic to this.
'''

@triton.jit
def fa_fwd_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, 
                    M, N, D: tl.constexpr, scale, causal: tl.constexpr,
                    stride_qm, stride_qd, stride_kn, stride_kd,
                    stride_vn, stride_vd, stride_om, stride_od,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # This will cater only to single head.
    # pass

    # How do you think the grid would be like?
    # one triton kernel for every query. so 1d grid.
    pid_m = tl.program_id(0)
    
    
    off_d = tl.arange(0, D)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_q= (off_m < M) & (off_d < D)

    off_q = off_m[:, None] * stride_qm + off_d[None, :] * stride_qd
    q_tile = tl.load(Q_ptr + off_q, mask = mask_q, other=0.0)   # [BLOCK_M, D]

    # We need softmax state tracking variables as well.

    # you do not need device argument because the kernel would anyway run inside 
    m_old = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_old = tl.zeros((BLOCK_M,), dtype=tl.float32) # this is the sum of exp values for each row.
    acc_old = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        # load k and v tiles at this point.
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_kv = (off_n < N) & (off_d < D) #THIS IS WRONG. THINK WHY WHY.
        off_k = off_d[:, None] * stride_kd + off_n[None, :] * stride_kn
        off_v = off_n[:, None] * stride_vn + off_d[None, :] * stride_vd

        # we want to load k in the transposed fashion.
        kt_tile = tl.load(K_ptr + off_k, mask=mask_kv, other=0.0)
        v_tile = tl.load(V_ptr + off_v, mask=mask_kv, other=0.0)

        scores = q_tile @ kt_tile
        scores = scores * scale

        if causal:
            # assert BLOCK_N == BLOCK_M, "Causal attention requires square blocks" # TODO: REMOVE THIS RESTRICTION
            
            k_idxs = start_n + tl.arange(0, BLOCK_N)
            # use offs_m as q_idxs
            mask_cas = k_idxs[None, :] <= offs_m[:, None]
            # scores.masked_fill(mask_cas, -float("inf"))
            scores = tl.where(mask_cas, scores, -float("inf"))

        m_new = tl.maximum(m_old, tl.max(scores, axis=-1))
        p_block = tl.exp(scores - m_new[:, None])
        l_block = tl.sum(p_block, axis=-1)
        acc_block = p_block @ v_tile

        alpha = tl.exp(m_old - m_new)
        l_new = alpha * l_old + l_block
        acc_new = alpha[:, None] * acc_old + acc_block

        m_old = m_new
        l_old = l_new
        acc_old = acc_new
    # out = acc_old / l_old[:, None]

    out = tl.where(l_old[:, None] > 0, acc_old / l_old[:, None], 0.0)

    out_off = off_m[:, None] * stride_om + off_d[None, :] * stride_od
    mask_off = off_m[:, None] < M & off_d[None, :] < D
    tl.store(O_ptr + out_off, out, mask = mask_off)


