import torch
import triton
import triton.language as tl


@triton.jit
def bias_add_kernel(
    x_ptr, b_ptr, output_ptr,
    M, N, stride_xm, stride_xn,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Какую строку и какой блок столбцов обрабатываем
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Индексы столбцов внутри этой строки
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs_n < N

    # x[pid_m, offs_n]
    x_offsets = pid_m * stride_xm + offs_n * stride_xn
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)

    # b[offs_n]: одинаковый bias для всех строк
    b = tl.load(b_ptr + offs_n, mask=mask, other=0.0)

    # Выход выделен как contiguous: stride строки = N
    out_offsets = pid_m * N + offs_n
    tl.store(output_ptr + out_offsets, x + b, mask=mask)


def bias_add(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    output = torch.empty(
        (M, N), device=x.device, dtype=x.dtype
    )

    if M == 0 or N == 0:
        return output

    BLOCK_SIZE_N = 128
    grid = (M, triton.cdiv(N, BLOCK_SIZE_N))

    bias_add_kernel[grid](
        x, b, output,
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return output
