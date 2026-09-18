'''
    check_correctness
    make_test_inputs_1d
    test_unary_op
    test_binary_op
    make_strided_input
    benchmark_unary
'''



import torch

# let's write pytorch layernorm


def layernorm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: int = 1e-5) -> torch.Tensor:

    B, N, D = X.shape

    # gamma shape will be 1, D so it will be brodcasted properly, right?


    # normalized dimensions will be the last, right?
    mean = torch.mean(X, dim=-1, keepdim=True)
    var = torch.var(X, dim=-1, unbiased=False, keepdim=True)

    out = X - mean
    out = out * torch.rqrt(var+eps)

    return out * gamma + beta


def softmax(X: torch.Tensor):

    B, N, D = X.shape

    max_t = torch.max(X, dim=-1, keepdim=True).values
    exp_t = torch.exp(X - max_t)
    exp_sum = torch.sum(exp_t, dim=-1, keepdim=True)

    return exp_t / exp_sum


    out = torch.zeros_like(X)
    out = torch.where(X>0, X, out)