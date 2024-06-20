#include <vector>
#include <Eigen/Dense>
#include <cuda_runtime.h>

#include "RandomInitialise.h"
#include "RandomSampleWrapper.hpp"

std::pair<std::vector<double>, double> GetBestRandomSample(
    const Eigen::MatrixXd& K,
    const std::vector<Eigen::VectorXd> S,
    double sg, double l, /* kernel params */
    std::vector<double>& yDiff,
    double a, /* explore vs exploit */
    const std::vector<double>& lb,
    const std::vector<double>& ubi,
    const int seed) {
    

    // converts to the lower level data structures
    // performs the work on the GPU
    // then returns the best value found for this batch

    // TODO - make sure not reusing the same random numbers

    // allocate device memory   
    
    // these params are tuned for a GeForce 2060
    // TODO - collect info automatically
    int warpSize = 32; // threads should be a multiple of this
    int batchSize = 3000; 
    int threadCount = warpSize * batchSize;

    float *V_d, *D_d, *W_d, *muPred_d, *sgPred_d, *innerMerit_d;
    float *S_d, *yDiff_d;
    float *K_d; // a matrix
    float *lb_d, *ub_d;
    float sg_d, l_d, a_d;    
    int Vstride, Dstride;

    Vstride = (int)lb.size();
    Dstride = (int)S.size();

    int size = threadCount * Vstride;

    cudaMalloc((void**)&V_d, size * sizeof(float));    

    fillRandomVectors(seed, V_d, size, 10000); // have each rng do 10000 gens        
    std::vector<float> h_V(size);

    cudaMemcpy(h_V.data(), V_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::vector<double> randomDoubles(size);

    for (int i = 0; i < size; i++) {
        randomDoubles[i] = static_cast<double>(h_V[i]);
    }

    cudaFree(V_d);

    return std::make_pair(randomDoubles, 0.0);
    
} 
