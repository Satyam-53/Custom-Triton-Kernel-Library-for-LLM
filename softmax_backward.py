import torch
import triton
import triton.language as tl

'''
y.shape = [M, N]
dy.shape = [M, N]
N <= BLOCK_N
y.is_contiguous()
dy.is_contiguous()

we'll make it tiled later on.

'''

@triton.jit
def softmax_backward_kernel(y_ptr, dy_ptr, dx_ptr, M, N, BLOCK_N: tl.constexpr):

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    y = tl.load(y_ptr + row * N +cols, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + row * N + cols, mask=mask, other=0.0)

    y_fp32 = y.to(tl.float32)
    dy_fp32 = dy.to(tl.float32)

    dot = tl.sum(y_fp32 * dy_fp32, axis = 0)

    dx = y_fp32 * (dyfp32 - dot)

    tl.store(dx_ptr + row * N + cols, dx, mask=mask)



# We can write the tiled version of the softmax backward as well.

@triton.jit
def softmax_backward_tiled(y_ptr, dy_ptr, dx_ptr, M, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)

    # cols = tl.arange(0, BLOCK_N)
    

    for start_n in range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        mask = cols < N

        offsets = pid * N + cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

        y_fp32 = y.to(tl.float32)

        dy_ptr = tl.load(dy_ptr + offsets, mask=mask, other=0.0)
        dy_fp32 = dy_ptr.to(tl.float32)

        dot += tl.sum(dy_fp32 * y_fp32, axis=0)

    for start_n in range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        mask = cols < N

        offsets = pid * N + cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        dy = tl.load(dy_ptr + offsets, mask=mask, other=0.0)

        y_fp32 = y.to(tl.float32)
        dy_fp32 = dy.to(tl.float32)

        dx = y_fp32(dy_fp32 - dot)
        tl.store(dx_ptr + offsets, dx, mask=mask)


# can have one version with the strided pattern as well, future todo.