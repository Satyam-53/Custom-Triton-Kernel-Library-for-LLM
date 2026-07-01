'''
approximate gelu and stable sigmoid time.
We can use tanh to approximate gelu. [IMPORTANT]

'''

import torch
import triton
import triton.language as tl

'''
    out = 0.5 * x ( 1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

'''

@triton.jit
def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # okay so the thing here is that, for production style, since we have lot
    # of very small numbers, we can have intermediates in fp32.
    x_fp32 = x.to(tl.float32)

    #  we refere to part inside tanh as inner.
    pi = 3.14159265358979
    inner = tl.sqrt(2 / pi) * (x_fp32 + 0.044715 * x_fp32 * x_fp32 * x_fp32)

    out = 0.5 * x_fp32 * (1 + tl.tanh(inner))
    out = out.to(x.dtype) # convert back to original dtype (fp16 or fp32) to avoid data corruption.

    tl.store(output_ptr + offsets, out, mask=mask)

def gelu(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:

    assert x.is_contiguous()

    out = torch.empty_like(x)   # since gelu is an element wise op.
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), ) #this needs to be tuple.

    gelu_kernel[grid](
        x,
        out,
        n_elements,
        BLOCK_SIZE = block_size
    )

    return out


# ref = torch.nn.functional.gelu(x, approximate="tanh")


# okay time for stable sigmoid.

@triton.jit
def stable_sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE:tl.constexpr):

    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    # stable sigmoid implementation

    pos = x_fp32 >= 0.0

    exp_neg = tl.exp(-x_fp32)
    exp_pos = tl.exp(x_fp32)

    # compute both the positive and negative cases
    out_pos = 1.0 / (1.0 + exp_neg)
    out_neg = exp_pos / (exp_pos + 1.0)

    out = tl.where(pos, out_pos, out_neg)
    out = out.to(x.dtype)

    # in the above implementation, we are paying twice the cost of exp() computation. We can do better.

    abs_x = tl.abs(x_fp32)
    e = tl.exp(-abs_x)

    out = tl.where(pos, 1.0 / 1.0 + e, e / (1.0 + e))   # this is more optimal.

    tl.store(out_ptr + offsets, out, mask=mask)

def stable_sigmoid(x: torch.Tensor, block_size : int = 1024):
    output = torch.empty_like(x)    # since it is an element wise op.
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), ) #for tuple.

    stable_sigmoid_kernel[grid](
        x, output, n_elements, BLOCK_SIZE = block_size
    )

    return output