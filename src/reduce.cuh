#pragma once

// Sum across a warp. Lane 0 ends up with the total.
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Sum across the whole block. EVERY thread gets the result.
// s must be a __shared__ array with at least 32 floats.
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
