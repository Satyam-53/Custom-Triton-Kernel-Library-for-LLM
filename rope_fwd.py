'''
We'll implement a rope kernel in this.
'''

import torch
import triton
import triton.language as tl

'''
x : [B, H, N, D]

the design for this kernel grid would be to tile along the sequence dimension.
Second dimension will be squash of B and H into one.

'''

@triton.jit
def rope_fwd_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, 
                        B, H, S, D: tl.constexpr,
                        stride_xb, stride_xh, stride_xs, stride_xd,
                        stride_coss, stride_cosd, stride_sins, stride_sind,
                        stride_outb, stride_outh, stride_outs, stride_outd,
                        BLOCK_S: tl.constexpr):

    pid_s = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh -  b*H

    x_base_ptr = x_ptr + b*stride_xb + h*stride_xh

    off_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    half_dim = D // 2   # this should be divisible by 2 properly.

    offs_d2 = tl.arange(0, half_dim)

    off_d_even = tl.arange(0, half_dim) * 2
    off_d_even = tl.arange(0, half_dim) * 2 + 1

    offs_x_even = off_s[:, None] * stride_xs + off_d_even[None, :] * offs_xd
    offs_x_odd = off_s[:, None] * stride_xs + off_d_odd[None, :] * stride_xd

    # what about the masks? 
    # we only need mask in the block s direction because that is where we are tiling.
    mask = offs_s[:, None] < S  # this needs to have the correct dimension.

    x_even = tl.load(x_ptr_base + offs_x_even, mask=mask, other=0.0)
    x_odd = tl.load(x_ptr_base + offs_x_odd, mask=mask, other=0.0)

    # time to load cos and sin
    off_cos = off_s[:, None] * stride_coss + off_d2[None, :] * stride_cosd
    off_sin = off_s[:, None] * stride_coss + off_d2[None, :] * stride_cosd

    cos = tl.load(cos_ptr_base + off_cos, mask=mask, other=0.0)
    sin = tl.load(cos_ptr_base + off_cos, mask=mask, other=0.0)

    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos

    out_base_ptr = O_ptr + b * stride_ob + h * stride_oh

    out_even_off = off_s[:, None] * stride_os + off_d_even[None, :] * stride_od
    out_odd_off = off_s[:, None] * stride_os + off_d_odd[None, :] * stride_od

    tl.store(out_base_ptr + out_even_off, y_even, mask=mask)
    tl.store(out_base_ptr + out_odd_off, y_odd, mask=mask)

def rope_fwd(x: torch.Tensor, block_s: int = 32):
    # now we have to apply rope tansitions to these tensors.
    # what if we generate the frequencies here itself?
    B, H, N, D = x.shape
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=x.device).float() / D))

    # how do we want to define grid in this case of ours?
    # so since it is applied along the feature dimension, we can squash the B, H and S
    # dimension. just need to figure out how to recover them back inside the kernel.
    # let's create the block_s, cos, sin and other things here onblu in this method.

    # theta = position * inv freq.
    theta = torch.arange(N, device=x.device).unsqueeze(1) * inv_freq.unsqueeze(0)

    cos = torch.cos(freqs).to(torch.float32)
    sin = torch.sin(freqs).to(torch.float32)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']), B*H)

    # cos and sin will be 2d only and have stride along seq and D dim only
    out = torch.empty_like(x)

    rope_fwd_kernel[grid](
        x, 
        cos, sin,
        out,

        B, H, N, D,

        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),

        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),

        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),

        BLOCK_S = block_s
    )

    return out.to(x.dtype)



# ######################################################################################
# Let's first write the pytorch reference for it.
# ######################################################################################

def rope(x, cos, sin):
    '''
    x:   [B, H, S, D]
    cos: [S, D // 2]
    sin: [S, D // 2]
    '''

    B, H, S, D = x.shape
    assert D % 2 == 0

    x_fp32 = x.float()

    x_even = x_fp32[..., 0::2]  #keep the rest of the indices as same and only skip the last one.
    x_odd = x_fp32[..., 1::2]

    cos = cos[None, None, :, :] # expand cos to match the size
    sin = sin[None, None, :, :]

    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos

    out = torch.empty_like(x_fp32)
    out[..., 0::2] = y_even
    out[..., 1::2] = y_odd

    return out.to(x.dtype)