'''
Q tile: [BLOCK_M, BLOCK_D]
K tile: [BLOCK_N, BLOCK_D]
'''

'''
S = Q @ K.T

Q.shape = [M, D]
K.shape = [N, D]
S.shape = [M, N]

'''

import torch
import triton
import triton.language as tl

@triton.jit
def score_matmul_kernel(Q_ptr, K_ptr, stride_qm, stride_qd, 
                            stride_kn, stride_kd, stride_cm, stride_cn,
                            M, D, N, BLOCK_M: tl.constexpr, 
                            BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr):

    # We'll have 2d program tiling in here. 
    pid_m = tl.program_id(0)    # for the output row block
    pid_n = tl.program_id(1)    # for the output column block

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_d in range(0, D, BLOCK_D):
        offs_d = start_d + tl.arange(0, BLOCK_D)
        # load Q and K tiles
        q_tile_off = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        mask_q = (offs_m[:, None] < M) & (offs_d[None, :] < D)
        
        k_tile_off = offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn  # this will load K directly as transposed.
        mask_k = (offs_n[None, :] < N) & (offs_d[:, None] < D)
 
        q = tl.load(Q_ptr + q_tile_off, mask=mask_q, other=0.0)
        k = tl.load(K_ptr + k_tile_off, mask=mask_k, other=0.0)

        # now how do we do the dot product?
        acc += tl.dot(q, k)
 
        
    # write back result
    off_c = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptr + off_c, acc, mask=mask_q & mask_n)




    # pass