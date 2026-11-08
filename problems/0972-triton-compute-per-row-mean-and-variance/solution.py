import torch
import triton
import triton.language as tl


@triton.jit
def mean_var_kernel(
    x_ptr, mean_ptr, var_ptr, M, N, stride_xm,
    BLOCK_SIZE_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    values = tl.load(
        x_ptr + row * stride_xm + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    mu = tl.sum(values, axis=0) / N

    diff = tl.where(mask, values - mu, 0.0)
    variance = tl.sum(diff * diff, axis=0) / N

    tl.store(mean_ptr + row, mu)
    tl.store(var_ptr + row, variance)


def mean_var(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape

    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    var = torch.empty(M, device=x.device, dtype=torch.float32)

    BLOCK_SIZE_N = triton.next_power_of_2(N)

    mean_var_kernel[(M,)](
        x, mean, var,
        M, N, x.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return mean, var