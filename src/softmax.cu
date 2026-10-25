#include "common.cuh"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

__global__ void softmax_v1(const float* x, float* y, int rows, int cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= rows){
        return;
    }
    const float* xr = x + row*cols;
    float* yr = y + row*cols;
    float m = -INFINITY;
    for(int j = 0;j<cols;j++){
        m = fmaxf(m,xr[j]);
    }
    float sum = 0.0f;
    for(int j = 0;j<cols;j++){
        sum += expf(xr[j] - m);
    }
    for (int j = 0; j < cols; j++) yr[j] = expf(xr[j] - m) / sum;
}

void softmax_cpu(const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* xr = x + (size_t)r * cols;
        float*       yr = y + (size_t)r * cols;

        double m = xr[0];
        for (int j = 1; j < cols; j++) m = std::max(m, (double)xr[j]);

        double sum = 0.0;
        for (int j = 0; j < cols; j++) sum += std::exp(xr[j] - m);

        for (int j = 0; j < cols; j++) yr[j] = (float)(std::exp(xr[j] - m) / sum);
    }
}

__global__ void softmax_v2(const float* x, float* y, int rows, int cols){
    __shared__ float s[256];
    int row = blockIdx.x;                   
    int tid = threadIdx.x;

    const float* xr = x + (size_t)row * cols;
    float*       yr = y + (size_t)row * cols;
    float m = -INFINITY;
    for(int j = tid;j<cols;j += blockDim.x){
        m = fmaxf(m,xr[j]);
    }
    s[tid] = m;
   __syncthreads();
    for(int stride = blockDim.x/2;stride> 0;stride /= 2){
        if(tid < stride){
            s[tid] = fmaxf(s[tid],s[tid + stride]);
        }
        __syncthreads();
    }
    m = s[0];
    __syncthreads();

    float sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) sum += expf(xr[j] - m);
    s[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    sum = s[0];     

    for(int j = tid;j<cols;j+=blockDim.x){
        yr[j] = expf(xr[j] - m) / sum;
    }
    

}

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


__device__ float block_reduce_max(float val, float* s) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

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


__device__ float block_reduce_sum(float val, float* s) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

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
__global__ void softmax_v3(const float* x, float* y, int rows, int cols) {
    __shared__ float smax[32];       
    __shared__ float ssum[32];        
    int row = blockIdx.x;
    int tid = threadIdx.x;

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

__device__ void merge_md(float& m, float& d, float m_other, float d_other){
    float m_new = fmaxf(m,m_other);
    if (m_new == -INFINITY) return; 
    d = d * expf(m - m_new) + d_other * expf(m_other - m_new);
    m = m_new;
}

__device__ void warp_reduce_md(float& m,float& d){
    for(int offset = 16;offset > 0;offset /= 2){
        float m_other = __shfl_down_sync(0xffffffff, m, offset);
        float d_other = __shfl_down_sync(0xffffffff, d, offset);
        merge_md(m, d, m_other, d_other);
    }
    
}

__global__ void softmax_v4(const float* x, float* y, int rows, int cols){
    __shared__ float sm[32];
    __shared__ float sd[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    const float* xr = x + (size_t)row * cols;
    float*       yr = y + (size_t)row * cols;
    float m = -INFINITY;
    float d = 0.0f;
    for(int j = tid;j<cols;j += blockDim.x){
        float v = xr[j];
        float m_new = fmaxf(m,v);
        d = d * expf(m - m_new) + expf(v - m_new);
        m = m_new;
    }

    warp_reduce_md(m, d);
    if(lane == 0){
        sm[warp] = m;
        sd[warp] = d;
    }
    __syncthreads();
    if(warp == 0){
        m = (lane < 8) ? sm[lane] : -INFINITY;
        d = (lane < 8) ? sd[lane] : 0.0f;
        warp_reduce_md(m, d);
        if (lane == 0) { sm[0] = m; sd[0] = d; }
    }
     __syncthreads();
    m = sm[0];
    d = sd[0];
    for (int j = tid; j < cols; j += blockDim.x) yr[j] = expf(xr[j] - m) / d;



}


void launch(int v,const float* d_x,float* d_y,int rows,int cols){
    switch(v){
        case 1: {
            int threads = 256;
            int blocks  = (rows + threads - 1) / threads;   // one thread per row
            softmax_v1<<<blocks, threads>>>(d_x, d_y, rows, cols);
            break;
           }
        case 2: {
            softmax_v2<<<rows, 256>>>(d_x, d_y, rows, cols);
            break;
        }
         case 3: {
            softmax_v3<<<rows, 256>>>(d_x, d_y, rows, cols);
            break;
        }
        case 4: {
            softmax_v4<<<rows, 256>>>(d_x, d_y, rows, cols);
            break;
        }
       default: std::cout << "Unknown version " << v << std::endl; exit(1);
    }
}

int main(){
    const int ROWS=4096, COLS=1024;
    size_t count = (size_t)ROWS * COLS;         
    size_t bytes = count * sizeof(float);  
    float* h_x   = (float*)malloc(bytes);
    float* h_y   = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    for (size_t i = 0; i < count; i++) h_x[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f;
    softmax_cpu(h_x, h_ref, ROWS, COLS);

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int v = 1; v <= 4; v++) {
        CUDA_CHECK(cudaMemset(d_y, 0, bytes));
        launch(v, d_x, d_y, ROWS, COLS);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

        double max_err = 0.0;
        for (size_t i = 0; i < count; i++) max_err = std::max(max_err, (double)fabsf(h_y[i] - h_ref[i]));

        int runs = 20;
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < runs; r++) launch(v, d_x, d_y, ROWS, COLS);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg = ms / runs;
        double gbps = 2.0 * bytes / (avg / 1000.0) / 1e9;   // minimum traffic: read x once + write y once

        std::cout << "v" << v << ": "
                  << (max_err < 1e-6 ? "correct" : "WRONG")
                  << " (max err " << std::scientific << std::setprecision(1) << max_err << ")   "
                  << std::fixed << std::setprecision(3) << avg << " ms   "
                  << std::setprecision(1) << gbps << " GB/s   ("
                  << std::setprecision(0) << 100.0 * gbps / 320.0 << "% of 320)"
                  << std::endl;
            }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x); free(h_y); free(h_ref);
    return 0;


        
    
    
}
