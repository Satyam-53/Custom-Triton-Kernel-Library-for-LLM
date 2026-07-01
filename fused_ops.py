import torch
import triton
import triton.language as tl

'''
FMA : Fused Multiply Add

The idea is to compute out = xi * yi + zi in one single kernel.
'''

@triton.jit
def fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr,n_elements, BLOCL_SIZE: tl.constexpr):

    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)

    out = x * y + z

    tl.store(out_ptr + offsets, out, mask=mask)

def fma(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, block_size: int):

    # assert for is_cuda(), is_contiguous(), "tensor must be on cuda and contiguous"
    assert x.shape == y.shape == z.shape, "tensors must have the same shape"

    output = torch.empty_like(x)

    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)   # this needs to be tuple for sure.

    fma_kernel[grid](x, y, z, output, n, BLOCK_SIZE=block_size)

    return output


'''
Residual Add 
The idea is to compute out = xi * yi + zi in one single kernel.
'''

# This should be very easy if you know the above implementation.


'''
IMPORTANT: Dropout-style masked residual
The idea is to compute outi = residual_i + mask_i * xi * scale_i in one single kernel.

scale = 1.0 / (1.0 - p),  p here will be the probability of dropout.
mask_i is either 0 or 1, where 1 means keep
'''

@triton.jit
def dropout_fused_kernel(x_ptr, residual_ptr, d_mask_ptr, output_ptr, 
                            n_elements, scale: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr):
    # check which of these needs tl.constexpr, right?
    pid = tl.program_id(0)

    # outi = residual_i + mask_i * x_i * scale

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    residual =  tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    d_mask = tl.load(d_mask_ptr + offsets, mask=mask, other=0.0)

    out = residual + d_mask * x * scale
    tl.store(output_ptr + offsets, out, mask=mask)

def dropout_fused():

    # we have to create the dropout mask inside this method.
    


