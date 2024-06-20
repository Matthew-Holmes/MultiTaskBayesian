#include "RandomInitialise.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void fillRandomVectorsKernel(
    int seed, float* V, const int size, const int threadStride) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i * threadStride >= size) {
        return;
    }

    curandState state;
    curand_init(seed, i, 0, &state); // each thread gets a difference sequence

    for (int j = 0; j != threadStride && (i * threadStride + j) < size; j++) {
        V[i * threadStride + j] = curand_uniform(&state);
    }
}


void fillRandomVectors(
    int seed, float* V, 
    const int size,  const int threadStride) {
    
    int blockSize = 256;
    int numThreads = (size + threadStride - 1) / threadStride;
    int numBlocks = (numThreads + blockSize - 1) / blockSize;

    // launch the kernel
    fillRandomVectorsKernel<<<numBlocks, blockSize>>>(
        seed, V, size, threadStride);

    cudaDeviceSynchronize();
}

 
