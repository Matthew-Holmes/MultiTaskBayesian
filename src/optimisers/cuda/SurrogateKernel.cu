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
    const float* lb, const float* ub,
    int ni) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i >= ni) { return; }

    // map the random values into the bounds 
    // since currently in the range [0,1]
    int vi = i * Vstride; // offset in concatenated sample space vectors
    for (int j = 0; j != Vstride; j++) {
        V[vi + j] = lb[j] + (ub[j] - lb[j]) * V[vi + j];
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
        for (int k = 0; k != Dstride; ++k) {
            wd += K[j*Dstride + k] * D[di + k];
        }
        W[di + j] = wd;
    }

    // compute muPred
    float mu_i = 0.0;
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

void computeInnerEvalations(
    float* V, int Vstride, /* random vecs        */
    float* D, int Dstride, /* distances to known */
    float* W,           /* weights, uses Dstride */
    float* muPred,      /* surrogate expectation */
    float* sgPred,      /* surrogate deviation   */
    float* innerMerit, /* want to minimise this */
    const float sg,     /* kernel deviation      */
    const float l,      /* kernel lengthscale    */
    const float* S,     /* samples, uses Dstride */
    const float* yDiff, /* shared across kernels */
    const float* K, /* inverse covariance matrix */
    const float a,  /* explore vs exploit coeff. */
    const float* lb, const float* ub, /*  bounds */
    int ni          /* number of random vectors  */
) {

    int blockSize = 256;
    int numBlocks = (ni + blockSize - 1) / blockSize;

    innerEvaluationsKernel<<<numBlocks, blockSize>>>(
        V, Vstride, D, Dstride, W, muPred, sgPred, innerMerit,
        sg, l, S, yDiff, K, a, lb, ub, ni);

    cudaDeviceSynchronize();

}




