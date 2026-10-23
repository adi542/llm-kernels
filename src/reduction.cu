#include "common.cuh"
#include <cmath>
#include <iostream>
#include <iomanip>


__global__ void reduce_v1(const float* in, float* out, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = s[0];
}


__global__ void reduce_v2(const float* in, float* out, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            s[index] += s[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = s[0];
}

__global__ void reduce_v3(const float* in, float* out, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = s[0];
}



__global__ void reduce_v4(const float* in, float* out, int n) {
    __shared__ float s[256];
    int tid = threadIdx.x;

    // step 1: each thread sums many numbers by itself (coalesced, no barriers)
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }

    // step 2: put the thread's sum in shared memory
    s[tid] = sum;
    __syncthreads();

    // step 3: the same tree as v3
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = s[0];
}



int launch(int v, const float* d_in, float* d_out, int n,int threads) {
    int blocks = (n + threads - 1) / threads; 
    switch (v) {
        case 1: reduce_v1<<<blocks, threads>>>(d_in, d_out, n); return blocks;
        case 2: reduce_v2<<<blocks, threads>>>(d_in, d_out, n); return blocks;
        case 3: reduce_v3<<<blocks, threads>>>(d_in, d_out, n); return blocks;
        case 4: blocks = 1024;                        
                reduce_v4<<<blocks, threads>>>(d_in, d_out, n); return blocks;
        default: std::cout << "Unknown version " << v << std::endl; exit(1);
    }
}


int main() {
    const int N       = 1 << 24;
    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;
    size_t bytes = (size_t)N * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)rand() / RAND_MAX;

    double ref = 0.0;
    for (int i = 0; i < N; i++) ref += h_in[i];

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, BLOCKS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    float* h_out = (float*)malloc(BLOCKS * sizeof(float));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int v = 1; v <= 4; v++) {                                       
        
        CUDA_CHECK(cudaMemset(d_out, 0, BLOCKS * sizeof(float)));         
        int used = launch(v, d_in, d_out, N,THREADS);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out, d_out, used * sizeof(float), cudaMemcpyDeviceToHost));
        double gpu = 0.0;
        for (int b = 0; b < used; b++) gpu += h_out[b];
        double rel_err = fabs(gpu - ref) / fabs(ref);

        
        int runs = 20;
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++) launch(v, d_in, d_out, N,THREADS);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg = ms / runs;
        double gbps = bytes / (avg / 1000.0) / 1e9;

       std::cout << "v" << v << ": "
                  << (rel_err < 1e-5 ? "correct" : "WRONG")
                  << " (rel err " << std::scientific << std::setprecision(1) << rel_err << ")   "
                  << std::fixed << std::setprecision(3) << avg << " ms   "
                  << std::setprecision(1) << gbps << " GB/s   ("
                  << std::setprecision(0) << 100.0 * gbps / 320.0 << "% of 320)"
                  << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    return 0;
}
