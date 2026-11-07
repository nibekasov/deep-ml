import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(
    x_ptr, output_ptr, M, N, stride_xm,
    BLOCK_SIZE_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)

    values = tl.load(
        x_ptr + row * stride_xm + cols,
        mask=cols < N,
        other=0.0,
    )

    total = tl.sum(values, axis=0)

    tl.store(output_ptr + row, total)


def row_sum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    output = torch.empty(M, device=x.device, dtype=x.dtype)

    if M == 0:
        return output

    BLOCK_SIZE_N = triton.next_power_of_2(max(N, 1))

    row_sum_kernel[(M,)](
        x, output, M, N, x.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return output