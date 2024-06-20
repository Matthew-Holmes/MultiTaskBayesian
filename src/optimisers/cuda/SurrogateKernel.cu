#include "SurrogateKernel.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>


// CUDA kernel
// TODO - constant arrays??
__global__ void computeKernel(
    float* V, int Vstride,
    float* D, int Dstride,
    float* W,
    float* muPred,
    float* sgPred,
    float* innerMerit,
    const float sg,
    const float l,
    const float* S,
    const float* yDiff,
    const float* K,
    const float a,
    const float* lb, const float* ub) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // map the random values into the bounds 
    // since currently in the range [0,1]
    int vi = i * Vstride;
    for (int j = 0; j != Vstride; j++) {
        V[vi + j] = lb[j] + (ub[j] - lb[j]) * V[vi + j]
    } 

    // fill Dstride distances using S and kernel params

    // compute Dstride weight values

    // compute muPred

    // compute sgPred

    // compute inner merit

}
