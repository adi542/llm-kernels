import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

// ---------- helpers: copied from softmax.cu
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
__device__ float block_reduce_sum(float val, float* s) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) s[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (lane < blockDim.x / 32) ? s[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) s[0] = val;
    }
    __syncthreads();
    return s[0];
}
__device__ float block_reduce_max(float val, float* s) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    val = warp_reduce_max(val);
    if (lane == 0) s[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (lane < blockDim.x / 32) ? s[lane] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0) s[0] = val;
    }
    __syncthreads();
    return s[0];
}

// ---------- kernel: your softmax v3, copied from softmax.cu
__global__ void softmax_v3(const float* x, float* y, int rows, int cols) {
    __shared__ float smax[32];
    __shared__ float ssum[32];
    int row = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + (size_t)row * cols;
    float*       yr = y + (size_t)row * cols;
    float m = -INFINITY;
    for (int j = tid; j < cols; j += blockDim.x) m = fmaxf(m, xr[j]);
    m = block_reduce_max(m, smax);
    float sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) sum += expf(xr[j] - m);
    sum = block_reduce_sum(sum, ssum);
    for (int j = tid; j < cols; j += blockDim.x) yr[j] = expf(xr[j] - m) / sum;
}

// ---------- translator: tensor in, tensor out
torch::Tensor softmax(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be on the GPU");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D: (rows, cols)");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int rows = x.size(0);
    int cols = x.size(1);
    auto y = torch::empty_like(x);

    softmax_v3<<<rows, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
    }

// ---------- kernel: your RMSNorm, copied from norms.cu
__global__ void rmsnorm(const float* x, const float* w, float* y, int rows, int cols, float eps) {
    __shared__ float s[32];
    int row = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + (size_t)row * cols;
    float*       yr = y + (size_t)row * cols;
    float ss = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) { float v = xr[j]; ss += v * v; }
    ss = block_reduce_sum(ss, s);
    float inv_rms = rsqrtf(ss / cols + eps);
    for (int j = tid; j < cols; j += blockDim.x) yr[j] = xr[j] * inv_rms * w[j];
}

// ---------- translator
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, double eps) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "x and w must be on the GPU");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && w.dtype() == torch::kFloat32, "must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D: (rows, cols)");
    TORCH_CHECK(w.dim() == 1 && w.size(0) == x.size(1), "w must have one value per column");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "must be contiguous");

    int rows = x.size(0);
    int cols = x.size(1);
    auto y = torch::empty_like(x);

    rmsnorm<<<rows, 256>>>(x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
                           rows, cols, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

'''

cpp_src = """
torch::Tensor softmax(torch::Tensor x);
torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, double eps);
"""

ext = load_inline(name="my_kernels", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["softmax", "rms_norm"], extra_cuda_cflags=["-O3"])

# ---------- test: compare with PyTorch's own softmax
x = torch.randn(4096, 1024, device="cuda")       # random numbers, on the GPU
y_mine  = ext.softmax(x)
y_torch = torch.softmax(x, dim=1)                # dim=1: softmax across each row

print("max difference:", (y_mine - y_torch).abs().max().item())

def time_ms(fn,runs=50):
    for i in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(runs):
        fn()
    end.record();
    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs

bytes_moved = 2 * x.numel() * 4   

ms = time_ms(lambda: ext.softmax(x))
print("mine :", round(ms, 3), "ms", round(bytes_moved / (ms / 1000) / 1e9, 1), "GB/s")

ms = time_ms(lambda: torch.softmax(x, dim=1))
print("torch:", round(ms, 3), "ms", round(bytes_moved / (ms / 1000) / 1e9, 1), "GB/s")

# ---------- RMSNorm: check against plain PyTorch, then race
eps = 1e-5
w = torch.rand(1024, device="cuda") + 0.5                     # one weight per column

def rms_torch(x, w, eps):                                     # RMSNorm in plain PyTorch
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps) * w

y_mine = ext.rms_norm(x, w, eps)
y_ref  = rms_torch(x, w, eps)
print("\nRMSNorm max difference:", (y_mine - y_ref).abs().max().item())

ms = time_ms(lambda: ext.rms_norm(x, w, eps))
print("mine         :", round(ms, 3), "ms", round(bytes_moved / (ms / 1000) / 1e9, 1), "GB/s")

ms = time_ms(lambda: rms_torch(x, w, eps))
print("plain PyTorch:", round(ms, 3), "ms", round(bytes_moved / (ms / 1000) / 1e9, 1), "GB/s")

if hasattr(torch.nn.functional, "rms_norm"):                  # only in newer PyTorch versions
    f = lambda: torch.nn.functional.rms_norm(x, (1024,), w, eps)
    ms = time_ms(f)
    print("F.rms_norm   :", round(ms, 3), "ms", round(bytes_moved / (ms / 1000) / 1e9, 1), "GB/s")
