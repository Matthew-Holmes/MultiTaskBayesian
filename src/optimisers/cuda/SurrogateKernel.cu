#include "SurrogateKernel.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>


// CUDA kernel
__global__ void innerEvaluationsKernel(
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
    int vi = i * Vstride; // offset in concatenated sample space vectors
    for (int j = 0; j != Vstride; j++) {
        V[vi + j] = lb[j] + (ub[j] - lb[j]) * V[vi + j]
    } 

    // fill Dstride distances using S and kernel params
    int di = i * Dstride; // offset in concatenated weight space vectors
    for (int j = 0; j != Dstride; ++j) {
        // compute Euclidean distance for sample j
        float dj = 0.0; 
        for (int k = 0; k != Vstride; ++k) {
            float djk =  S[j * Vstride + k] - V[vi + k];
            dj += djk * djk;
        }        
        dj = sqrtf(dj); // euclidean distance
        dj = sg * sg * expf( (-1.0 * dj) / (2.0 * l * l));
        D[di + j] = dj;
    }

    // compute Dstride weight values, matrix vector multiplication
    for (int j = 0; j != Dstride; ++j) {
        float wd = 0.0;
        for (int k = 0; k != DStride; ++k) {
            wd += K[j*Dstride + k] * D[di + k];
        }
        W[di + j] = wd;
    }

    // compute muPred
    float mu_i = 0.0
    for (int j = 0; j != Dstride; ++j) {
        mu_i += W[di + j] * W[di + j];    
    }
    muPred[i] = mu_i;
    
    // compute sgPred
    float dot_i= 0.0;
    for (int j = 0; j != Dstride; ++j) {
        dot_i += W[di + j] * D[di + j];
    }
    sgPred[i] = sg - sqrtf(dot_i);
    
    // compute inner merit, using explore vs exploit coefficient
    innerMerit[i] = muPred[i] - a * sgPred[i];

}
