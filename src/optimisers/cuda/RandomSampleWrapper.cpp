#include <vector>
#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "RandomInitialise.h"
#include "SurrogateKernel.h"
#include "RandomSampleWrapper.hpp"

std::pair<std::vector<double>, double> GetBestRandomSample(
    const Matrix& K,
    const std::vector<Eigen::VectorXd> S,
    double sg, double l, /* kernel params */
    std::vector<double>& yDiff,
    double a, /* explore vs exploit */
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    const int seed) {
    

    // converts to the lower level data structures
    // performs the work on the GPU
    // then returns the best value found for this batch

    // TODO - make sure not reusing the same random numbers

    // allocate device memory   
    
    // these params are tuned for a GeForce 2060
    // TODO - collect info automatically
    // TODO - update according to the number of samples, since don't
    // want to exceed total device memory
    int warpSize = 32; // threads should be a multiple of this
    int batchSize = 3000; 
    const int threadCount = warpSize * batchSize;

    // each thread is responsible for one random sample surrogate
    // merit function evaluation

    float *V_d, *D_d, *W_d, *muPred_d, *sgPred_d, *innerMerit_d;
    float *S_d, *yDiff_d;
    float *K_d; // a matrix
    float *lb_d, *ub_d;
    int Vstride, Dstride;

    Vstride = (int)lb.size();
    Dstride = (int)S.size();

    // ________________________________________________________________________
    // ****************** DEVICE MEMORY ALLOCATION ****************************
    // ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

    // concatentated vectors in search space
    cudaMalloc((void**)&V_d, Vstride * threadCount * sizeof(float));    
    // concatenated vectors in distance/weights space
    cudaMalloc((void**)&D_d, Dstride * threadCount * sizeof(float));
    cudaMalloc((void**)&W_d, Dstride * threadCount * sizeof(float));
    // concatenated predicted values
    cudaMalloc((void**)&muPred_d,      threadCount * sizeof(float));
    cudaMalloc((void**)&sgPred_d,      threadCount * sizeof(float));
    cudaMalloc((void**)&innerMerit_d,  threadCount * sizeof(float));

    // copies of values passed to functions, for device access
    cudaMalloc((void**)&S_d,     Vstride * Dstride * sizeof(float));
    cudaMalloc((void**)&yDiff_d, Dstride           * sizeof(float));
    cudaMalloc((void**)&K_d,     Dstride * Dstride * sizeof(float));
    cudaMalloc((void**)&lb_d,    Vstride           * sizeof(float));
    cudaMalloc((void**)&ub_d,    Vstride           * sizeof(float));

    // ________________________________________________________________________
    // ********************** TRANSFER INPUT DATA *****************************
    // ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

    std::vector<float> S_flattened;
    for (const auto& vec: S) {
        for (int i = 0; i < vec.size(); ++i) {
            S_flattened.push_back(static_cast<float>(vec[i]));
        }
    } 
    cudaMemcpy(S_d, S_flattened.data(), S_flattened.size() * sizeof(float), 
        cudaMemcpyHostToDevice);
    
    int Kdim = K.rows(); // K is square
    std::vector<float> K_flattened(Kdim * Kdim);
    for (int i = 0; i < Kdim; ++i) {
        for (int j = 0; j < Kdim; ++j) {
            K_flattened.push_back(static_cast<float>(K(/*row*/i,/*col*/j)));
        }
    }
    cudaMemcpy(K_d, K_flattened.data(), K_flattened.size() * sizeof(float),
        cudaMemcpyHostToDevice);

    std::vector<float> ub_float, lb_float, yDiff_float;

    for (decltype(lb.size()) i = 0; i != lb.size(); ++i) {
        lb_float.push_back(static_cast<float>(lb[i]));
        ub_float.push_back(static_cast<float>(ub[i]));
    }   

    for (decltype(yDiff.size()) i = 0; i != yDiff.size(); ++i) {
        yDiff_float.push_back(static_cast<float>(yDiff[i]));
    }

    cudaMemcpy(lb_d, lb_float.data(),       Vstride * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(ub_d, ub_float.data(),       Vstride * sizeof(float), 
        cudaMemcpyHostToDevice);
    cudaMemcpy(yDiff_d, yDiff_float.data(), Dstride * sizeof(float),
        cudaMemcpyHostToDevice);

    // ________________________________________________________________________
    // ******************** RANDOM VECTOR GENERATION **************************
    // ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
        
    // use a separate kernel for the random allocation
    // since these spin up an RNG state, so don't want one per random vector
    // since these have low dimensionality
    
    int size = threadCount * Vstride;

    fillRandomVectors(seed, V_d, size, 10000); // have each rng do 10000 gens        
    // ________________________________________________________________________
    // ***************** SURROGATE MERIT COMPUTATION **************************
    // ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    
    computeInnerEvaluations(V_d, Vstride, D_d, Dstride, W_d, muPred_d, sgPred_d,
        innerMerit_d, sg, l, S_d, yDiff_d, K_d, a, lb_d, ub_d,
        threadCount);
    

    // _________________________________________________________________________
    // ****************** FIND BEST SURROGATE EVALUATION ***********************
    // ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

    // TODO - reduce using another kernel to find the minimum innerMerit_d value
    // won't implement this from the outset since this step may be insignificant
    // in overall timings 

    int bestIndex = 0; double bestSMerit = innerMerit_d[0];
    for (int i = 0; i != threadCount; ++i) {
        if (innerMerit_d[i] < bestSMerit) {
            bestIndex = i;
            bestSMerit = innerMerit_d[i];
        }
    }

    std::vector<double> bestVec;
    
    for (int i = 0; i != Vstride; ++i) {
        bestVec.push_back(static_cast<double>(V_d[bestIndex * Vstride + i]));
    } 
    
    // ________________________________________________________________________
    // *********************** CLEAN UP AND RETURN ****************************
    // ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾

    cudaFree(V_d);
    cudaFree(D_d);
    cudaFree(W_d);
    cudaFree(muPred_d);
    cudaFree(sgPred_d);
    cudaFree(innerMerit_d);
    cudaFree(S_d);
    cudaFree(K_d);
    cudaFree(lb_d);
    cudaFree(ub_d);
    cudaFree(yDiff_d);

    return std::make_pair(bestVec, static_cast<double>(bestSMerit));
    
} 
