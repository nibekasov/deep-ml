#include <cuda_runtime.h>
#include <vector>

__global__ void index_kernel(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] =i;
    }
}

std::vector<int> global_thread_indices(int n) {
    if (n <= 0) {
        return {};
    }

    std::vector<int> result(n);
    int* device_out = nullptr;
    size_t bytes = static_cast<size_t>(n) * sizeof(int);

    cudaMalloc(reinterpret_cast<void**>(&device_out), bytes);

    int threads_per_block = 256;
    int num_blocks = 1 + (n-1) / threads_per_block;

    index_kernel<<<num_blocks, threads_per_block>>>(device_out, n);

    cudaMemcpyAsync(
        result.data(),
        device_out,
        bytes,
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_out);
    return result;
}
