'''
So now we will implement batched multi head FA.

q: [B, H, M, D]
k: [B, H, N, D]
v: [B, H, N, D]
o: [B, H, M, D]
'''

import torch
import triton
import triton.language as tl

@triton.jit
def batched_mhfa_fwd_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, 
                                scale, B, H, M, N, D: tl.constexpr,
                                strid_qb, strid_qh, strid_qm, strid_qd,
                                strid_kb, strid_kh, strid_kn, strid_kd,
                                strid_vb, strid_vh, strid_vn, strid_vd,
                                strid_ob, strid_oh, strid_on, strid_od,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                causal: tl.constexpr = False):
    # We need a 2d grid, and the second dim here will be
    # the squashed batch and HEAD into one and we know how
    # to get each using the div and reminder trick.

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b*H

    q_base_ptr = Q_ptr + b * stride_qb + h * stride_qh
    k_base_ptr = K_ptr + b * stride_kb + h * stride_kh
    v_base_ptr = V_ptr + b * stride_vb + h * stride_vh

    offs_d = tl.arange(0, D)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    offs_q = offs_m[:, NONE] * stride_qm + offs_d[None, :] * stride_qd
    mask_q = (offs_m[:, NONE] < M) &(offs_d[None, :] < D)

    q_tile = tl.load(q_base_ptr + offs_q, mask=mask_q, other=0.0)

    # online state tracking variables.
    m_old = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_old = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_old = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        offs_kt = offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
        mask_kt = (offs_d[:, None] < D) & (offs_n[None, :] < N)
        kt_tile = tl.load(k_base_ptr + offs_kt, mask=mask_kt, other=0.0)

        offs_v = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        mask_v = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        v_tile = tl.load(v_base_ptr + offs_v, mask=mask_v, other=0.0)

        scores = tl.dot(q_tile, kt_tile)  # [BLOCK_M, BLOCK_N]

        # generate a score mask here.
        valid_m = offs_m < M
        valid_n = offs_n < N

        scores_mask = valid_m[:, None] & valid_n[:, None]

        if causal:
            # generate a causal mask here.
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            score_mask = score_mask & causal_mask

        scores = tl.where(scores_mask, scores, -float("inf"))

        m_new = tl.maximum(m_old, tl.max(scores, axis=-1))
        # m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)

        p_block = tl.exp(scores - m_new[:, None])
        l_block = tl.sum(p_block, axis=-1)
        acc_block = tl.dot(p_block, v_tile)

        alpha = tl.exp(m_old - m_new)
        l_new = alpha * l_old + l_block
        acc_new = alpha[:, None] * acc_old + acc_block

        # Time to update the state
        m_old = m_new
        l_old = l_new
        acc_old = acc_new
    
    out = tl.where(l_old[:, None] > 0, acc_old / l_old[:, None], 0.0)

    o_base_ptr = O_ptr + b*stride_ob + h*stride_oh
    out_offs = offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    mask_offs = mask_q

    tl.store(o_base_ptr + out_offs, out, mask=mask_offs)


# Let's now write the wrapper for it.
def batched_mhfa_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
                        block_m: int = 64, block_n: int = 64):

    B, H, M, D = Q.shape
    N = K.shape(2)

    scale = D ** -0.5

    output = torch.empty((M, D), dtype=Q.dtype, device=Q.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), B*H)

    batched_mhfa_fwd_kernel[grid](
        Q, K, V, scale, B, H, M, N, D
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        BLOCK_M = block_m, BLOCK_N = block_n
    )

    return output


        

