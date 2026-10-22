#include "common.cuh"
#include <cmath>
#include <iostream>

__global__ void reduce_v1(const float* in,float* out,int n){
    __shared__ float s[256];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + tid;
    s[tid] = (i <n) ? in[i] : 0.0f;
    __syncthreads();

    for(int stride = 1;stride<blockDim.x;stride *= 2){
        if(tid % (2 * stride) == 0){
            s[tid] += s[tid + stride];
         }
      __syncthreads();
    }

  if (tid == 0){
      out[blockIdx.x] = s[0];
  } 
  }


int main(){
    const int N       = 1 << 24;   
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS -1)/THREADS;
    size_t bytes = sizeof(float) * size_t(N);

    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)rand() / RAND_MAX;

    double ref = 0.0;
    for (int i = 0; i < N; i++) ref += h_in[i];
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, BLOCKS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    reduce_v1<<<BLOCKS, THREADS>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_out = (float*)malloc(BLOCKS * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, BLOCKS * sizeof(float), cudaMemcpyDeviceToHost));

    double gpu = 0.0;
    for (int b = 0; b < BLOCKS; b++) gpu += h_out[b];
    double rel_err = fabs(gpu - ref) / fabs(ref);

    std::cout << "v1: GPU = " << gpu 
              << "  CPU = " << ref 
              << "  relative error = " << rel_err 
              << "  " << (rel_err < 1e-5 ? "correct" : "WRONG") << std::endl;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    int runs = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < runs; r++) {
        reduce_v1<<<BLOCKS, THREADS>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg = ms / runs;
    double gbps = bytes / (avg / 1000.0) / 1e9;

    std::cout << "v1: " << avg << " ms  " 
              << gbps << " GB/s  (" 
              << (100.0 * gbps / 320.0) << "% of 320)" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    return 0;
    

    

    
    
}
