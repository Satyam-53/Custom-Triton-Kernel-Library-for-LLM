import torch
import triton
import triton.language as tl


'''
so till now we assumed and asserted that the tensors would be contiguous but now
we are moving twoards strided tensors.

x = base[::2]
x logically has elements adjacent but not physically.
so instead of x_ptr + offsets, we need

x_ptr + offsets * stride_x

logical_index = offsets
physical_index = logical_index * stride

'''

'''
In this file, we'll implement strided ReLU, vector add, and affine transform.
'''

# now we won't create output tensors as torch.empty_like, rather with torch.empty only.

@triton.jit
def strided_relu_kernel():
    pass