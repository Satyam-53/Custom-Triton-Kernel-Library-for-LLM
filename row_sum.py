'''
so this is a row sum kernel implementation where each
program id will handle a single row.

'''

import torch
import triton
import triton.language as tl

@triton.jit
def row_sum_kernel(x_ptr, stride_x, out_ptr, row_size, ROWS: tl.constexpr):
    # each triton program id will be handling a single row
    pid = tl.program_id(0) # which row are we computing?

    offsets = pid * stride_x + tl.arange(0, row_size)
    
    x = tl.load(x_ptr + offsets) # load the row data
    row_sum = tl.sum(x, axis=0) # get one valued tensor of the sum of the row
    tl.store(out_ptr + pid, row_sum) # store the result in the output tensor


def row_sum(x: torch.Tensor) -> torch.Tensor:

    rows = x.shape[0]
    cols = x.shape[1]
    stride_x = x.stride(0) # how many elements to skip to get to the next row?

    out = torch.empty(rows, dtype=x.dtype, device=x.device)

    grid = lmabda meta: (meta["ROWS"], )

    row_sum_kernel[grid](x, out, cols, ROWS=rows)
    return out
