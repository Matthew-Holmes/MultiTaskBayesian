#include "InitCuda.h"
#include <cuda_runtime.h>

// dummy CUDA kernel
__global__ void dummyKernel() {
    // do nothing
}

void initCuda() {
    dummyKernel<<<1,1>>>();
    cudaDeviceSynchronize();
}
