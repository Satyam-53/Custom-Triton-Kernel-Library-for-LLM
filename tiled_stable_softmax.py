'''
Here we will write a tiled stable softmax kernel, in a streamed manner.
This will do only 2 pass. one pass for numerator, second pass for denominator.
'''

import torch
import triton
import triton.language as tl

@triton.jit
def tiled_stable_softmax_kernle(x_ptr, out_ptr, stride_m, stride_n, M, N, BLOCK_N: tl.constexpr, ):
    pid = tl.program_id(0) # this is our row.

    # we need to have variable to save the state.
    m_old = float('-inf')
    l_old = 0.0

    for start_n in range(0, N, BLOCK_N):
        offsets = pid * stride_m + stride_n * (start_n + tl.arange(0, BLOCK_N)) # offset into the row, with block size step
        # mask = offsets < N     --> this is wrong, mask should be on column indices only.

        x_tile = tl.load(x_ptr+offsets, mask=mask, other=float('-inf'))

        m_new = tl.max(x_tile, axis=0)
        m_new = tl.maximum(m_old, m_new) # new max value of the row. 

        alpha = tl.exp(m_old - m_new)
        
        x_tile = tl.exp(x_tile - m_new) # stable softmax computation. 
        l_new = alpha * l_old + tl.sum(x_tile, axis=0)

        # store the output of x_tile in the correcut out_ptr values.
        # if I have to do the second pass, why do I store now??
        # tl.store(out_ptr + offsets, x_tile, mask=mask)

        m_old = m_new
        l_old = l_new

    # now do the second part of the algorithm. 
    m_final = m_old
    l_final = l_old

    for start_n in range(0, N, BLOCK_N):
        offsets = pid * stride_m + (start_n + tl.arange(0, BLOCK_N)) * stride_n
        mask = offsets < N

        x_tile = tl.load(x_ptr+offsets, mask=mask, other=float('-inf'))
        x_tile = tl.exp(x_tile - m_final) / l_final # normalize with final values. 

        tl.store(out_ptr + offsets, x_tile, mask=mask) # store the output of x_tile in the correct out_ptr 

def tiled_stable_softmax(x: torch.Tensor, block_n: int = 32) -> torch.Tensor:
    assert x.dim()==2 # only supporting 2D tile for now.

    M = x.shape[0]
    N = x.shape[1]
    stride_m = x.stride(0)
    stride_n = x.stride(1)

    grid = (M, )    # each program handling one row.

    # create out same as x, and use stride info to store values
    out = torch.empty_like(x)

    tiled_stable_softmax_kernle[grid](
        x, out, stride_m, stride_n, M, N, BLOCK_N=block_n
    )

    return out
