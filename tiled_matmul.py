'''
Implement C = A @ B using tiled blocks, tl.dot, fp32 accumulation, and tail masks.
'''

import torch
import triton
import triton.language as tl


'''
A.shape = [M, K]
B.shape = [K, N]
C.shape = [M, N]

The way we'll write this kernel is that, we would assign each block to a tile of the output matrix C. 
BLOCK_M, BLOCK_N will be computed by a single triton kernel. If we want to use BLOCK_K as well, this
would be something similar to the streaming version.
'''

@triton.jit
def tiled_matmul_kernel(A_ptr, B_ptr, C_ptr,
                        M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    
    # so we'll be having the 2d program instances.
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)

    # these would be from matrix A, the multiplication with K would be the equivalent of the stride.
    row_offsets = ((row_id * BLOCK_M) + tl.arange(0, BLOCK_M)) 
    col_offsets = ((col_id * BLOCK_N) + tl.arange(0, BLOCK_N))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tlfloat32)

    for start_k in range(0, K, BLOCK_K):
        # load a tile of A and B, and compute the partial dot product
        # we need to create 2d offset of the tiles to load
        k_idxs = start_k + tl.arange(0, BLOCK_K)

        a_tile_off = row_offsets[:, None] * K + k_idxs[None, :]
        b_tile_off = k_idxs[:, None] * N + col_offsets[None, :]

        a_mask = (row_offsets[:, None] < M) & (k_idxs[None, :] < K)
        b_mask = (col_offsets[None, :] < N) & (k_idxs[None, :] < K)

        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    # now store all this in a C matrix
    c_off = row_offsets[:, None] * N + col_offsets[None, :]
    c_mask = 

    tl.store()
