import torch
import triton
import triton.language as tl

'''
x.shape = [M, N]
gamma.shape = [N]
beta.shape = [N]

'''

@triton.jit
def layernorm_fwd_kernel(x_ptr, y_ptr, wPtr, b_ptr,
                        stride_o, N, eps, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)  #which row we are processing
    cols = tl.arange(0, BLOCK_SIZE)  #what
    mask = cols < N

    x = tl.load(x_ptr + pid * stride_o + cols, mask = mask, other=0.0)  #load the row of x
    
    mean = tl.sum(x, axis=0) / N  # since we have just one row loaded.
    

    # load w and b
    w = tl.load(wPtr + cols, mask=mask, other=0.0)
    b = tl.load(bPtr + cols, mask=mask, other=0.0)

    x_centred = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centred**2, axis=0) / N
    rstd = tl.rsqrt(var + eps)  #this is the reciprocal of standard deviation.

    y = w * (x - mean) * rstd + b

    tl.store(y_ptr + pid * stride_o + cols, y, mask=mask)

    # pass
def layernorm_fwd(x, weight, bias, eps=1e-5):
    M, N = x.shape  #assuming only 2d tensors.

    y = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (M,) #this needs to be a tuple.

    layernorm_fwd_kernel[grid](
        x, y, weight, bias, x.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE
    )

    return y