import torch
import triton
import triton.language as tl


##################################################################################
# RELU Kernel
##################################################################################

@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr+offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)

    tl.store(out_ptr+offsets, out, mask=mask)
    # pass


# wrapper function that takes two tensors and returns their element-wise sum
def relu(x: torch.Tensor, block_size:int = 1024) -> torch.Tensor:
    # pass
    # do some checks over here
    assert x.is_cuda()
    assert x.is_contiguous()
    # create output tensor
    out = torch.empty_like(x)
    # get the number of elements in the tensor
    n = x.numel()
    # define the grid  

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]))

    relu_kernel[grid](x, out, n, BLOCK_SIZE=block_size)

    return out

# test method


##################################################################################
# Square Kernel
##################################################################################

# exactly same as relu kernel but with square operation instead of relu
# out = x * x instead of tl.maximum()

##################################################################################
# Sigmoid Kernel (this will be interesting)
##################################################################################

# for naive implementatin, just the operation will change
# out = 1.0 / (1.0 + tl.exp(-x))

@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x_ptr+offsets, mask=mask, other=0.0)

    out = 1.0 / (1.0 + tl.exp(-x))

    tl.store(out_ptr+offsets, out, mask=mask)
    # pass

def sigmoid(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    # TODO: implement sigmoid using a Triton kernel
    assert x.is_cuda()
    assert x.is_contiguous()

    out = torch.empty_like(x)   #since elementwise operations preserve shape, we can use empty_like
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    # TODO: fill in the rest of the implementation
    sigmoid_kernel[grid](x, out, n, BLOCK_SIZE=block_size)
    return out


##################################################################################
# Tanh Kernel (this will be interesting)
##################################################################################
# if you use tl.tanh then is can also be simply implemented in the same way as above kernels, 
# apploximation is used under the hood for tanh operator.

# for reference use torch.tanh()


# time for reusable test function...
def test_unary_kernels(name, triton_fn, torch_fn, rtol=1e-2, atol=1e-2):
    # pass
    sizes= [1, 3, 5, 7, 8,]
    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        for size in sizes:
            x = torch.randn(size, dtype=dtype, device='cuda')
            
            ref = torch_fn(X)
            tri = triton_fn(x)

            torch.testing.assert_close(ref, tri, rtol=rtol, atol=atol)
            print(f"{name} test passed for size {size} and dtype {dtype}")
    print(f"{name} passed")

test_unary_op("tanh", tanh_kernel, torch.tanh)
# test_unary_op("square", square_kernel, lambda x: x*x)



