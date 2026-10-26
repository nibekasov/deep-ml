#include <cuda_runtime.h>
#include <vector>

__global__ void relu_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = fmaxf(0.0f, x[i]);
    }
}

std::vector<float> relu(const std::vector<float>& x) {
    if (x.empty()) {
        return {};
    }

    int n = static_cast<int>(x.size());
    size_t bytes = x.size() * sizeof(float);

    float* d_x = nullptr;
    float* d_out = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_x), bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_out), bytes);

    cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads; 

    relu_kernel<<<blocks, threads>>>(d_x, d_out, n);
    cudaDeviceSynchronize(); 

    std::vector<float> result(n);
    cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_out);

    return result;
}

