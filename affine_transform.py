# Affine transformations are linear transformation followed by translation.
'''
A neural network layer often computes:
y=Wx+by = Wx + by=Wx+b
This is called an affine layer or affine transform.
The name comes from the fact that it's a combination of a linear transformation (Wx) and a translation (b).
'''

import torch
import triton
import triton.language as tl

# if we assume our x to be 1d, then it is going to be same as any other
# elementwise 1d kernel.

@triton.jit
def affine_transform_kernel(x_ptr, a, b, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    p_id = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    out = a * x + b
    tl.store(output_ptr + offsets, out, mask=mask)

def affine_transform(x: torch.tensor, a: float, b: float, block_size: int = 1024) -> torch.tensor:

    assert x.is_cuda()
    assert x.is_contiguous()

    out = torch.empty_like(x)
    n = x.numel()

    grid = lambda meta : (triton.cdiv(n, meta["BLOCK_SIZE"]),)  # extra comma to make it a tuple.
    affine_transform_kernel[grid](
        x, a, b, out, n, BLOCK_SIZE=block_size
    )

    return out

