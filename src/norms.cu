#include "common.cuh"
#include "reduce.cuh"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>


__global__ void rmsnorm(const float* x, const float* w, float* y,
                        int rows, int cols, float eps) {
    __shared__ float s[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* xr = x + (size_t)row * cols;
    float*       yr = y + (size_t)row * cols;

    float ss = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float v = xr[j];
        ss += v * v;
    }
    ss = block_reduce_sum(ss, s);

    float inv_rms = rsqrtf(ss / cols + eps);

    for (int j = tid; j < cols; j += blockDim.x) {
        yr[j] = xr[j] * inv_rms * w[j];
    }
}


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


int main() {
    const int ROWS = 4096, COLS = 1024;
    const float EPS = 1e-5f;
    size_t count = (size_t)ROWS * COLS;
    size_t bytes = count * sizeof(float);

    // inputs: x in [-1, 1), weights in [0.5, 1.5)
    float* h_x   = (float*)malloc(bytes);
    float* h_w   = (float*)malloc(COLS * sizeof(float));
    float* h_y   = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    for (size_t i = 0; i < count; i++) h_x[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int j = 0; j < COLS; j++)      h_w[j] = (float)rand() / RAND_MAX + 0.5f;
    rmsnorm_cpu(h_x, h_w, h_ref, ROWS, COLS, EPS);

    float *d_x, *d_w, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_w, COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w, COLS * sizeof(float), cudaMemcpyHostToDevice));

   
    CUDA_CHECK(cudaMemset(d_y, 0, bytes));
    rmsnorm<<<ROWS, 256>>>(d_x, d_w, d_y, ROWS, COLS, EPS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (size_t i = 0; i < count; i++) {
        if (std::isnan(h_y[i])) { max_err = INFINITY; break; }
        max_err = std::max(max_err, (double)fabsf(h_y[i] - h_ref[i]));
    }

    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    int runs = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) rmsnorm<<<ROWS, 256>>>(d_x, d_w, d_y, ROWS, COLS, EPS);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg = ms / runs;
    double gbps = 2.0 * bytes / (avg / 1000.0) / 1e9;   // read x once + write y once (w is tiny)

    std::cout << "rmsnorm: "
              << (max_err < 1e-5 ? "correct" : "WRONG")
              << " (max err " << std::scientific << std::setprecision(1) << max_err << ")   "
              << std::fixed << std::setprecision(3) << avg << " ms   "
              << std::setprecision(1) << gbps << " GB/s   ("
              << std::setprecision(0) << 100.0 * gbps / 320.0 << "% of 320)"
              << std::endl;

   
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_y));
    free(h_x); free(h_w); free(h_y); free(h_ref);
    return 0;
}
