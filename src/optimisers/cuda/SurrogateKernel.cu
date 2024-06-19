#include "SurrogateKernel.h"
#include <cuda_runtime.h>

// CUDA kernel
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
    const float a) {

// randomly populate V

// fill Dstride distances using S and kernel params

// compute Dstride weight values

// compute muPred

// compute sgPred

// compute inner merit

}
