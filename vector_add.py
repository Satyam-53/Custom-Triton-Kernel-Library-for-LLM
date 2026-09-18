import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    p_id = tl.program_id(0)
    offset = p_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) # this is the offset for each thread in the block. 0, 1
    mask = offset < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    out = x + y
    tl.store(output_ptr + offsets, out, mask=mask)

def vector_add(x: torch.tensor, y: torch.tensor, block_size: int = 1024) -> torch.tensor:
    # TODO: implement vector add using a Triton kernel

    # things to check here, on same device, same number of elements, and also contiguous
    assert x.is_cuda() and y.is_cuda()
    # maybe assert the device as well?
    assert x.numel() == y.numel()
    assert x.is_contiguous() 
    assert y.is_contiguous()

    out = torch.empty_like(x)
    n = x.numel()

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]))    # this will be used during autotuning.

    vector_add_kernel[grid](
        x, y, out, n, BLOCK_SIZE=block_size

    )

    return out


def test_vector_add():
    sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    dtypes = [torch.float32, torch.float16]

    for dtype in dtypes:
        for n in sizes:
            x = torch.randn(n, device="cuda", dtype=dtype)
            
            y = torch.randn(n, device="cuda", dtype=dtype)

            ref = x+y
            tri = vector_add(x, y)

            torch.testing.assert_close(tri, ref, rtol=1e-2, atol=1e-2)

            max_abs_Err = (tri - ref).abs.max().item()
            print(f"Max absolute error for {n} elements: {max_abs_Err}")
    
    print("All vector add tests passed")

test_vector_add()