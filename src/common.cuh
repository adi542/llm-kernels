#pragma once
#include <cstdio>
#include <cstdlib>

// Stops the program if a CUDA call fails, printing where and why
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                              \
        }                                                         \
    } while (0)
