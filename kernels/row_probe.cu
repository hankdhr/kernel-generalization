// Handwritten instrumentation probes, not agent outputs or competitive baselines.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <climits>

template<int BLOCK>
__global__ void row_probe_kernel(const float* x, float* y, int64_t cols,
                                 int64_t s0, int64_t s1, int op) {
    __shared__ float scratch[BLOCK];
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    float value = op == 5 ? -INFINITY : 0.0f;
    for (int64_t col = tid; col < cols; col += BLOCK) {
        const float v = x[row * s0 + col * s1];
        if (op == 5) value = fmaxf(value, v);
        else if (op == 1 || op == 4) value += v * v;
        else if (op == 2) value += fmaxf(v, 0.0f);
        else value += v;
    }
    scratch[tid] = value;
    __syncthreads();
    for (int step = BLOCK / 2; step > 0; step /= 2) {
        if (tid < step) scratch[tid] = op == 5
            ? fmaxf(scratch[tid], scratch[tid + step])
            : scratch[tid] + scratch[tid + step];
        __syncthreads();
    }
    const float reduction = scratch[0];
    if (op < 4) {
        if (tid == 0) y[row] = op == 3 ? reduction / cols : reduction;
        return;
    }
    if (op == 4) {
        const float scale = rsqrtf(reduction / cols + 1e-5f);
        for (int64_t col = tid; col < cols; col += BLOCK)
            y[row * cols + col] = x[row * s0 + col * s1] * scale;
        return;
    }
    // Every thread has read the shared maximum before scratch is reused.
    __syncthreads();
    value = 0.0f;
    for (int64_t col = tid; col < cols; col += BLOCK)
        value += expf(x[row * s0 + col * s1] - reduction);
    scratch[tid] = value;
    __syncthreads();
    for (int step = BLOCK / 2; step > 0; step /= 2) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        __syncthreads();
    }
    const float denominator = scratch[0];
    for (int64_t col = tid; col < cols; col += BLOCK)
        y[row * cols + col] = expf(x[row * s0 + col * s1] - reduction) / denominator;
}

torch::Tensor row_probe(torch::Tensor x, int64_t op, int64_t block) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 && x.dim() == 2,
                "expected a CUDA float32 matrix");
    TORCH_CHECK(x.size(0) > 0 && x.size(0) <= INT_MAX && x.size(1) > 0, "invalid shape");
    TORCH_CHECK(x.stride(0) > 0 && x.stride(1) > 0, "positive strides required");
    TORCH_CHECK(op >= 0 && op <= 5, "invalid operation");
    TORCH_CHECK(block == 128 || block == 256, "block must be 128 or 256");
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = op < 4 ? torch::empty({x.size(0)}, x.options())
                    : torch::empty({x.size(0), x.size(1)}, x.options());
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    if (block == 128)
        row_probe_kernel<128><<<x.size(0), 128, 0, stream.stream()>>>(
            x.data_ptr<float>(), y.data_ptr<float>(), x.size(1), x.stride(0), x.stride(1), op);
    else
        row_probe_kernel<256><<<x.size(0), 256, 0, stream.stream()>>>(
            x.data_ptr<float>(), y.data_ptr<float>(), x.size(1), x.stride(0), x.stride(1), op);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_probe", &row_probe, "Handwritten row operation probe");
}
