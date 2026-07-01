import torch
import math

'''
We’ll implement:

online_attention_forward_torch(q, k, v, block_n)

for single-head 2D tensors:

q: [M, D]
k: [N, D]
v: [N, D]
out: [M, D]

This prototype should match:

torch.softmax(q @ k.T * scale, dim=-1) @ v
'''

# Reference Attn
def attention_ref(q, k, v, causal=False):
    M, D = q.shape
    N, _ = k.shape

    scale = D ** -0.5

    scores = q.float() @ k.float().T
    scores = score * scale

    if causal:
        assert M == N
        mask = torch.triu((M, N), device=q.device, dtype=torch.bool,
        diagonal=1, )

        scores = scores.masked_fill(mask, -float("inf"))

    p = torch.softmax(scores, dim=-1)
    out = p @ v.float()

    return out.to(q.dtype)

# online attn prototype
def online_attn_forward_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                block_n: int = 32, causal=False) -> torch.Tensor:

    assert q.dim() == k.dim() == v.dim() == 2

    M , D = q.shape
    N, Dk = k.shape
    Nv, Dv = v.shape

    assert D == Dk
    assert D == Dv
    assert N == Nv

    if causal:
        assert M == N

    # so here we are basically assuming we are able to load the whole q.
    # let's take care of some precision.
    q_fp32 = q.to(torch.float32)
    k_fp32 = k.to(torch.float32)
    v_fp32 = v.to(torch.float32)

    scale = D ** -0.5
    q_idx = torch.arange(M, device=q.device)

    # If we have to do online attention, make sure we create state tarcking variables


    m_old = torch.full((M,), -float("inf"), device=q.device, dtype=torch.float32) # [BLOCK_M, None] -> [M, None] but we only need the first BLOCK_
    l_old = torch.zeros((M,), device=q.device, dtype=torch.float32)
    acc_old = torch.zeros((M, D), device=q.device, dtype=torch.float32)


    for start_n in range (0, N, block_n):
        # load k and v blocks, assuming the D dimension will completely fit.
        k_tile = k_fp32[start_n : start_n + block_n, :] # [block_n, D]
        v_tile = v_fp32[start_n : start_n + block_n, :]

        scores = q_fp32 @ k_tile.T # [M, block_n]
        scores *= scale

        if causal:
            # create a mask
            k_idx = torch.arange(start_n, start_n + block_n, device=q.device)
            mask = k_idx[None, :] > q_idx[:, None]
            scores = scores.masked_fill(mask, -float("inf"))
        
        m_block = torch.max(scores, dim=-1).values
        m_new = torch.maximum(m_old, m_block)   # 1d tensor only.
        p_block = torch.exp(scores - m_new[:, None])
        l_block = torch.sum(p_block, dim=-1)
        acc_block = p_block @ v_tile # [M, D]


        alpha = torch.exp(m_old - m_new)
        l_new = alpha * l_old + l_block

        acc_new = alpha[:, None] * acc_old + acc_block
       
        # update the running stats
        m_old = m_new
        l_old = l_new
        acc_old = acc_new

    out = acc_old / l_old[:, None]
    
    # In case of full masked rwos, there can be cases where l_old is zero. This will cause a division by zero error.
    out = torch.where(l_old[:, None] > 0,
            acc_old/ l_old[:, None], torch.zeros_like(acc_old))

    return out.to(q.dtype)
