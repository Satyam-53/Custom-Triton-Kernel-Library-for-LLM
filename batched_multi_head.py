'''
Q.shape = [B, H, M, D]
K.shape = [B, H, N, D]
S.shape = [B, H, M, N]

'''

import torch
import triton
import triton.language as tl

'''
So we'll be writing the batched multi head attention.
'''

@triton.jit
def batched_multi_head_attn_kernel(Q_ptr, K_ptr, S_ptr,
                                    stride_qb, stride_qh, stride_qm, stride_qd,
                                    stride_kb, stride_kh, stride_kn, stride_kd,
                                    stride_sb, stride_sh, stride_sm, stride_sn,
                                    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
                                    scale, B, H, M, D, N):

    # grid will still be  3d now, the other dimension will tell me which bh pair are we executing on.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    batch = pid_bh // B
    head = pid_bh - batch*B

    # each program instance would be responsibel for computing a block of S, correspoinding to
    # a specific B, H, pair.
    q_ptr_base = Q_ptr + batch * stride_qb + head * stride_qh
    k_ptr_base = K_ptr + batch * stride_kb + head * stride_kh
    s_ptr_base = S_ptr + batch * stride_sb + head * stride_sh

    # now normal matmul with a transposed caveat.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=float32)

    for start_d in range(0, D, BLOCK_D):
        idxs_d = start_d + offs_d

        q_tile_off = offs_m[:, None] * stride_qm + idxs_d[None, :] * stride_qd
        mask_q = (offs_m[:. None] < M) & (idxs_d[None, :] < D)

        # we'll load this in a transposed manner itself.
        k_tile_off = idxs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
        mask_k = (idxs_d[:, None] < D) & (offs_n[None, :] < N)

        q_tile = tl.load(q_ptr_base + q_tile_off, mask=mask_q, other=0.0)
        kt_tile = tl.load(k_ptr_base + k_tile_off, mask=mask_k, other=0.0)

        # compute attention scores with matmul
        acc += tl.dot(q_tile, kt_tile)

    acc = acc * scale

    s_tile_off = offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    mask_s = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(s_ptr_base + s_tile_off, acc, mask=mask_s)

# wrapper method.
def batched_multi_head_attn(Q: torch.Tensor, K: torch.Tensor):
    assert Q.shape[-1] == K.shape[-1], "Q and K must have the same last dimension"

    B, H, M, D = Q.shape
    N = K.shape[-2]

    S = torch.empty((M, N), device='cuda', dtype=torch.float32)

    stride_qb, stride_qh, stride_qm, stride_qd = Q.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = K.stride()
    stride_sb, stride_sh, stride_sm, stride_sn = S.stride()

    # create 3d grid.
    grid = ((triton.cdiv(M, meta[])), (), B*H)








