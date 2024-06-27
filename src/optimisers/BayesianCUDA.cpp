#include "BayesianCUDA.hpp"
#include "OptimisationPolicy.hpp"
#include "RandomSampleWrapper.hpp"

#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <string>
#include <iterator>
#include <iostream>

#include <float.h>

using std::vector;
using clock_hr = std::chrono::high_resolution_clock;
// typdef Eigen::MatrixXD Matrix // in header file

// helper function just for this compilation unit
double static Timems(decltype(clock_hr::now()) start, decltype(clock_hr::now()) end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

BayesianCUDA::BayesianCUDA() : warmUpGPU(true) {}

vector<double> BayesianCUDA::optimise(
    double &bestMeritOut,
    const FunctionBase& meritFunction,
    const vector<double>& lb,
    const vector<double>& ub,
    int maxit, int timePerItms) const {

    std::size_t dim = lb.size();

    // notation: meritFunction is f : R^n -> R
    // notation: f(x) =  y
    // notation: f evalulation number: i
    // notation: f(xi) = yi

    vector<Eigen::VectorXd> xis;
    vector<double> yis;

    // burn in for longer when in higher dimensions
    // need to avoid a degenerate burn in geometry
    int burnIn = 8 + dim;

    if (maxit <= burnIn) {
        // low maxit edge case
        DoBurnIn(meritFunction, lb, ub, xis, yis, maxit);
        return GetBestEval(bestMeritOut, xis, yis); 
    }
        
    DoBurnIn(meritFunction, lb, ub, xis, yis, burnIn); 

    int it = burnIn + 1; // can assume at least one iteration remaining 

    // setup the optimisation policy
    OptimisationPolicy policy;
    policy.SetInnerOptimisationTimeAllocation(timePerItms);
    policy.SetMinInnerLoopEvals(1); // NOTE - each eval does 32,000
    // evals since uses GPU

    while (it <= maxit) {

        auto start = clock_hr::now();

            DoBayesianStep(meritFunction, lb, ub, xis, yis, policy);

        auto end   = clock_hr::now();

        policy.InformFullIterationTime(it, Timems(start, end)); it++;
    }
    policy.SaveInteractionLog("logs.txt");
    warmUpGPU = true; // so no hysteresis
    
    return GetBestEval(bestMeritOut, xis, yis); 
}


void BayesianCUDA::DoBurnIn(
    const FunctionBase& meritFunction,
    const vector<double>& lb,
    const vector<double>& ub,
    vector<Eigen::VectorXd>& xis,
    vector<double>& yis,
    int numSamples) const {
    
    std::size_t dim = lb.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); 
    
    xis.reserve(numSamples);
    yis.reserve(numSamples);    

    for (int i = 0; i < numSamples; ++i) {
        Eigen::VectorXd x(dim);
        for (std::size_t j = 0; j < dim; ++j) {
            x(j) = lb[j] + dis(gen) * (ub[j] - lb[j]);
        }
        double y = meritFunction.eval(
            vector<double>(x.data(), x.data() + x.size()));
        xis.push_back(x);
        yis.push_back(y);
    }
}

vector<double> BayesianCUDA::GetBestEval(
    double &bestMeritOut,
    const vector<Eigen::VectorXd>& xis,
    const vector<double>& yis) const {

    auto minElementIt = std::min_element(yis.begin(), yis.end());
    bestMeritOut = *minElementIt;

    int minIndex = std::distance(yis.begin(), minElementIt);

    Eigen::VectorXd bestVector = xis[minIndex];
    return vector<double>(
        bestVector.data(), bestVector.data() + bestVector.size());
}

void BayesianCUDA::DoBayesianStep(
    const FunctionBase& meritFunction,
    const vector<double>& lb,
    const vector<double>& ub,
    vector<Eigen::VectorXd>& xis,
    vector<double>& yis_raw,
    OptimisationPolicy& policy) const {

    // Notation: average of recorded values - mu
    // Notation: sqrt(variance of recorded) - sigma (sg)

    // transform the data so more suitable for Bayesian inference
    vector<double> yis = yis_raw;

    double min_yi = *std::min_element(yis.begin(), yis.end());
    double shift = 1 - min_yi;
    for (double& yi : yis) {
        yi += shift;
    }

    for (double& yi : yis) {
        yi = std::log(yi);
    }

    double mu = std::accumulate(yis.begin(), yis.end(), 0.0) / yis.size();
    double sg = SampleDev(yis, mu);

    double ls = 0.1; // length scale
    ls *= sqrt((double)lb.size());

//    std::cout << ls << std::endl;

    // we'll automate length scale finding later, now use 0.4 as default

    vector<Eigen::VectorXd> samples; 
    vector<double> sampleMerits;
    vector<bool> mask;    
    bool doMask = false;
    int maxSamples;

    if (policy.KnowMaxMeritSamplesToUse()) {
        maxSamples = policy.GetMaxMeritSamplesToUse();
        if (maxSamples < (int)xis.size()) {
            doMask = true;
        }
    }

    if (doMask) {
        mask = GenerateRandomMask((int)xis.size(), maxSamples);
        for (int i = 0; i != (int)xis.size(); ++i) {
            if (mask[i]) {
                samples.push_back(xis[i]);
                sampleMerits.push_back(yis[i]);
            }
        }
    } else {
        // what to do if copying here is too much overhead??
        samples = xis;
        sampleMerits = yis;
    }
    
    // create the matrix and record the time 
    auto start = clock_hr::now();
        Matrix cov = ComputeCovarianceMatrix(samples, sg, ls);
        Matrix K = cov.inverse();
    auto end = clock_hr::now();

    double matrixTimems = Timems(start, end);
    policy.Inform((int)samples.size(), matrixTimems);

    Eigen::Map<const Eigen::VectorXd> y(
        sampleMerits.data(), sampleMerits.size());

    Eigen::VectorXd yDiff = y - Eigen::VectorXd::Constant(y.size(), mu); 

    vector<double> yDiff_std(yDiff.data(), yDiff.data() + yDiff.size());
    
    int numEvals = 1; // number of GPU Monte-Carlo passes

    bool knowEvalNumber = policy.KnowEvalsToDo((int)samples.size());

    if (knowEvalNumber) {
        numEvals = policy.GetInnerLoopEvalsToPerform((int)samples.size());
    }
    
    // used to preserve the best GPU random sample
    std::vector<double> bestVec;
    double bestSMerit = DBL_MAX;

    if (warmUpGPU) {
        // the first use of the GPU has big overhead we don't want
        // to register in the policy, so do a throwaway call to get
        // the synchronisation context etc. going
        auto [localBestVec, localBestSMerit] = GetBestRandomSample(
            K, samples, sg, ls, yDiff_std, 3.0, lb, ub, 0);
        warmUpGPU = false;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 4.0);

    double a = dis(gen);

    auto evalStart = clock_hr::now();

        // CUDA wrapper call
        // gets surrogate merit and coresponding sample vector
        for (int j = 0; j < numEvals; ++j) {
            auto [localBestVec, localBestSMerit] = GetBestRandomSample(
                K, samples, sg, ls, yDiff_std, a, lb, ub, rd());
                // use random device to avoid seed collisions

            if (localBestSMerit < bestSMerit) {
                bestSMerit = localBestSMerit;
                bestVec = localBestVec;
            }
        }

    auto evalEnd = clock_hr::now();

    double evalsTimems = Timems(evalStart, evalEnd);
    policy.Inform((int)samples.size(), numEvals, evalsTimems);

    if (!knowEvalNumber) {
        // will decide here how many more to do
        double remaining = policy.InnerOptimisationTimeAllocation();
        remaining -= matrixTimems;
        remaining -= evalsTimems;
        remaining *= 0.95; // redundancy
                           // won't double register with policy
                           // since we pass eval time too when informing
        
        double estTimePerEval = evalsTimems / numEvals;
        int toDo = remaining / estTimePerEval;
        
        auto evalStart2 = clock_hr::now();
            
            // more CUDA wrapper calls
            for (int j = 0; j < toDo; ++j) {
                auto [localBestVec, localBestSMerit] = GetBestRandomSample(
                    K, samples, sg, ls, yDiff_std, a, lb, ub, rd());
                // +10 in case did the first few passes to get the timings' data

                if (localBestSMerit < bestSMerit) {
                    bestSMerit = localBestSMerit;
                    bestVec = localBestVec;
                }
            }

        auto evalEnd2 = clock_hr::now();
        policy.Inform((int)samples.size(), toDo, Timems(evalStart2, evalEnd2));
    }

    // eval meritFunction and update xis, yis
    Eigen::Map<Eigen::VectorXd> ev(&bestVec[0], bestVec.size());
    xis.push_back(ev);
    yis_raw.push_back(meritFunction.eval(bestVec));
}

vector<bool> BayesianCUDA::GenerateRandomMask(
    int totalSize, int samples) const {
    vector<bool> mask(totalSize, false);

    std::fill(mask.begin(), mask.begin() + samples, true);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(mask.begin(), mask.end(), gen);

    return mask;
}
 
double BayesianCUDA::SampleDev(const vector<double>& v, double mu) const {
    double sum_sq_diff = 0.0;
    for (double val : v) {
        double diff = val - mu;
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff / (v.size() - 1));
} 

Matrix BayesianCUDA::ComputeCovarianceMatrix(
    const vector<Eigen::VectorXd>& xis,
    double sigma,
    double lengthScale) const {

    size_t n = xis.size();
    Matrix covarianceMatrix(n,n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            covarianceMatrix(i,j) = Kernel(
                xis[i], xis[j], sigma, lengthScale);   
            if (i == j) {
                covarianceMatrix(i,j) += sigma * sigma * 0.1; // nugget
            }
        }
    }

    return covarianceMatrix;
} 


double BayesianCUDA::Kernel(
    const Eigen::VectorXd& lhs,
    const Eigen::VectorXd& rhs,
    double sigma, double lengthScale) const {
    
    double sum = (lhs - rhs).squaredNorm();  
    return sigma*sigma*std::exp(-sum / (2.0 * lengthScale * lengthScale));
}
