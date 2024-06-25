#ifndef SURROGATEKERNEL_H
#define SURROGATEKERNEL_H

void computeInnerEvalations(
    float* V, int Vstride, /* random vecs        */
    float* D, int Dstride, /* distances to known */
    float* W,           /* weights, uses Dstride */
    float* muPred,      /* surrogate expectation */
    float* sgPred,      /* surrogate deviation   */
    float* innerMerit, /* want to minimise this  */
    const float sg,     /* kernel deviation      */
    const float l,      /* kernel lengthscale    */
    const float* S,     /* samples, uses Dstride */
    const float* yDiff, /* shared across kernels */
    const float* K, /* inverse covariance matrix */
    const float a,  /* explore vs exploit coeff. */
    const float* lb, const float* ub, /*  bounds */
    int ni          /* number of random vectors  */
);

#endif
