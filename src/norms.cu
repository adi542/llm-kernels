#include "common.cuh"
#include "reduce.cuh"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

// ======================================================== kernels
// RMSNorm: one block per row, one reduction (sum of squares)
__global__ void rmsnorm(const float* x, const float* w, float* y,
                        int rows, int cols, float eps) {
    __shared__ float s[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * cols;
    float*       yr = y + (size_t)row * cols;

    float ss = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) { float v = xr[j]; ss += v * v; }
    ss = block_reduce_sum(ss, s);

    float inv_rms = rsqrtf(ss / cols + eps);
    for (int j = tid; j < cols; j += blockDim.x) yr[j] = xr[j] * inv_rms * w[j];
}

// LayerNorm: one block per row, two reductions (mean, then variance)
__global__ void layernorm(const float* x, const float* gamma, const float* beta, float* y,
                          int rows, int cols, float eps) {
    __shared__ float s1[32];
    __shared__ float s2[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * cols;
    float*       yr = y + (size_t)row * cols;

    float sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) sum += xr[j];
    float mean = block_reduce_sum(sum, s1) / cols;

    float sq = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) { float dlt = xr[j] - mean; sq += dlt * dlt; }
    float var     = block_reduce_sum(sq, s2) / cols;
    float inv_std = rsqrtf(var + eps);

    for (int j = tid; j < cols; j += blockDim.x)
        yr[j] = (xr[j] - mean) * inv_std * gamma[j] + beta[j];
}

// ======================================================== CPU references (double precision)
void rmsnorm_cpu(const float* x, const float* w, float* y, int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * cols;
        float*       yr = y + (size_t)r * cols;
        double ss = 0.0;
        for (int j = 0; j < cols; j++) ss += (double)xr[j] * xr[j];
        double inv_rms = 1.0 / std::sqrt(ss / cols + eps);
        for (int j = 0; j < cols; j++) yr[j] = (float)(xr[j] * inv_rms * w[j]);
    }
}

void layernorm_cpu(const float* x, const float* gamma, const float* beta, float* y,
                   int rows, int cols, float eps) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * cols;
        float*       yr = y + (size_t)r * cols;
        double sum = 0.0;
        for (int j = 0; j < cols; j++) sum += xr[j];
        double mean = sum / cols;
        double sq = 0.0;
        for (int j = 0; j < cols; j++) { double dlt = xr[j] - mean; sq += dlt * dlt; }
        double inv_std = 1.0 / std::sqrt(sq / cols + eps);
        for (int j = 0; j < cols; j++) yr[j] = (float)((xr[j] - mean) * inv_std * gamma[j] + beta[j]);
    }
}

// ======================================================== launcher
// which = 1: RMSNorm (uses w as its weight)   which = 2: LayerNorm (w = gamma, b = beta)
void launch(int which, const float* d_x, const float* d_w, const float* d_b, float* d_y,
            int rows, int cols, float eps) {
    switch (which) {
        case 1: rmsnorm<<<rows, 256>>>(d_x, d_w, d_y, rows, cols, eps); break;
        case 2: layernorm<<<rows, 256>>>(d_x, d_w, d_b, d_y, rows, cols, eps); break;
        default: std::cout << "Unknown kernel " << which << std::endl; exit(1);
    }
}

// ======================================================== main
int main() {
    const int ROWS = 4096, COLS = 1024;
    const float EPS = 1e-5f;
    size_t count = (size_t)ROWS * COLS;
    size_t bytes = count * sizeof(float);

    // inputs: x in [-1, 1), weights in [0.5, 1.5), biases in [-0.5, 0.5)
    float* h_x = (float*)malloc(bytes);
    float* h_w = (float*)malloc(COLS * sizeof(float));
    float* h_b = (float*)malloc(COLS * sizeof(float));
    float* h_y = (float*)malloc(bytes);
    float* h_ref_rms = (float*)malloc(bytes);
    float* h_ref_ln  = (float*)malloc(bytes);
    for (size_t i = 0; i < count; i++) h_x[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int j = 0; j < COLS; j++) {
        h_w[j] = (float)rand() / RAND_MAX + 0.5f;
        h_b[j] = (float)rand() / RAND_MAX - 0.5f;
    }
    rmsnorm_cpu(h_x, h_w, h_ref_rms, ROWS, COLS, EPS);
    layernorm_cpu(h_x, h_w, h_b, h_ref_ln, ROWS, COLS, EPS);

    float *d_x, *d_w, *d_b, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_w, COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, COLS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, COLS * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const char* names[] = {"", "rmsnorm", "layernorm"};

    for (int which = 1; which <= 2; which++) {
        float* h_ref = (which == 1) ? h_ref_rms : h_ref_ln;

        // ---- run once (warmup), then check
        CUDA_CHECK(cudaMemset(d_y, 0, bytes));
        launch(which, d_x, d_w, d_b, d_y, ROWS, COLS, EPS);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

        double max_err = 0.0;
        for (size_t i = 0; i < count; i++) {
            if (std::isnan(h_y[i])) { max_err = INFINITY; break; }
            max_err = std::max(max_err, (double)fabsf(h_y[i] - h_ref[i]));
        }

        // ---- time 20 runs
        int runs = 20;
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++) launch(which, d_x, d_w, d_b, d_y, ROWS, COLS, EPS);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg = ms / runs;
        double gbps = 2.0 * bytes / (avg / 1000.0) / 1e9;

        std::cout << names[which] << ": "
                  << (max_err < 1e-5 ? "correct" : "WRONG")
                  << " (max err " << std::scientific << std::setprecision(1) << max_err << ")   "
                  << std::fixed << std::setprecision(3) << avg << " ms   "
                  << std::setprecision(1) << gbps << " GB/s   ("
                  << std::setprecision(0) << 100.0 * gbps / 320.0 << "% of 320)"
                  << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_y));
    free(h_x); free(h_w); free(h_b); free(h_y); free(h_ref_rms); free(h_ref_ln);
    return 0;
}
