#include "Bayesian.hpp"
#include "matplotlib-cpp.h"
#include <random>
#include <vector>
#include <algorithm>
#include <string>
#include <iterator>

namespace plt = matplotlibcpp;

std::vector<double> Bayesian::optimise(
    double &bestMeritOut,
    const FunctionBase& meritFunction,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    int maxit) const {

    std::size_t dim = lb.size();

    // notation: meritFunction is f : R^n -> R
    // notation: f(x) =  y
    // notation: f evalulation number: i
    // notation: f(xi) = yi

    std::vector<Eigen::VectorXd> xis;
    std::vector<double> yis;

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

    while (it <= maxit) {
        DoBayesianStep(meritFunction, lb, ub, xis, yis);
        it++;
    }

    return GetBestEval(bestMeritOut, xis, yis); 
}


void Bayesian::DoBurnIn(
    const FunctionBase& meritFunction,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    std::vector<Eigen::VectorXd>& xis,
    std::vector<double>& yis,
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
            std::vector<double>(x.data(), x.data() + x.size()));
        xis.push_back(x);
        yis.push_back(y);
    }
}


std::vector<double> Bayesian::GetBestEval(
    double &bestMeritOut,
    const std::vector<Eigen::VectorXd>& xis,
    const std::vector<double>& yis) const {

    auto minElementIt = std::min_element(yis.begin(), yis.end());
    bestMeritOut = *minElementIt;

    int minIndex = std::distance(yis.begin(), minElementIt);

    Eigen::VectorXd bestVector = xis[minIndex];
    return std::vector<double>(
        bestVector.data(), bestVector.data() + bestVector.size());
}

void Bayesian::DoBayesianStep(
    const FunctionBase& meritFunction,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    std::vector<Eigen::VectorXd>& xis,
    std::vector<double>& yis) const {

    // Notation: average of recorded values - mu
    // Notation: sqrt(variance of recorded) - sigma (sg)

    double mu = std::accumulate(yis.begin(), yis.end(), 0.0) / yis.size();
    double sg = SampleDev(yis, mu);    

    double ls = 0.4; // length scale

    // we'll automate length scale finding later, now use 0.4 as default
    Matrix cov = ComputeCovarianceMatrix(xis, sg, ls);
    Matrix K = cov.inverse();

    Eigen::Map<const Eigen::VectorXd> y(yis.data(), yis.size());

    // produce a lambda surrogate function for mu_pred, sg_pred
    auto surrogate = [&] (const Eigen::VectorXd xp) {

        Eigen::VectorXd   dists(xis.size());
        Eigen::VectorXd weights(xis.size());

        for (std::size_t i = 0; i < xis.size(); ++i) {
            dists[i] = Kernel(xp, xis[i], sg, ls);
        }

        weights = K * dists;
        
        double mu_pred = weights.dot(y);
        double sg_pred = sg - weights.dot(dists);
        
        return std::make_pair(mu_pred, sg_pred);
    };
        
    // optimise that (random sample)

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); 

    std::size_t dim = lb.size();
    int numSamples = 1000; // will tune later
    std::vector<Eigen::VectorXd> sampleVecs; sampleVecs.reserve(numSamples);

    std::vector<double> sampleVals, sampleMus, sampleSgs;
    sampleVals.reserve(numSamples);
    sampleMus.reserve(numSamples);
    sampleSgs.reserve(numSamples);

    for (int j = 0; j < 1000; ++j) {
        Eigen::VectorXd x(dim);
        for (std::size_t k = 0; k < dim; ++k) {
            x[k] = lb[k] + dis(gen) * (ub[k] - lb[k]);        
        }
        auto [mu_pred, sg_pred] = surrogate(x);
        sampleMus.push_back(mu_pred);
        sampleSgs.push_back(sg_pred);

        sampleVecs.push_back(x);
        sampleVals.push_back(mu_pred - 1.0 * sg_pred);
        // later we'll randomise the explore/exploit tradeoff
    }

    auto minElementIt = std::min_element(
        sampleVals.begin(), sampleVals.end());
    int minIndex = std::distance(sampleVals.begin(), minElementIt);
    Eigen::VectorXd testVec = sampleVecs[minIndex];  

    // plots for debugging
    if (dim == 1) {
        std::size_t it = xis.size();
        Plot1D(sampleVecs, sampleMus,
             "mus" + std::to_string((int)it) +".png");   
        Plot1D(sampleVecs, sampleSgs,
             "sgs" + std::to_string((int)it) +".png");    
        Plot1D(sampleVecs, sampleVals,
             "merit" + std::to_string((int)it) +".png");    
    } else if (dim == 2) {
        std::size_t it = xis.size();
        Plot2D(sampleVecs, sampleMus,
             "mus" + std::to_string((int)it) +".png");   
        Plot2D(sampleVecs, sampleSgs,
             "sgs" + std::to_string((int)it) +".png");    
        Plot2D(sampleVecs, sampleVals,
             "merit" + std::to_string((int)it) +".png");    
    }
    // eval meritFunction and update xis, yis
    xis.push_back(testVec);
    yis.push_back(meritFunction.eval(
        std::vector<double>(testVec.data(), testVec.data() + testVec.size()))); 
}
        
void Bayesian::Plot1D(
    const std::vector<Eigen::VectorXd>& xs,
    const std::vector<double>& ys,
    const std::string& filename) const {
 
    std::vector<std::pair<double,double>> data;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        data.push_back(std::make_pair(xs[i][0], ys[i]));
    }
    std::sort(data.begin(), data.end());

    std::vector<double> sortedX(xs.size());
    std::vector<double> sortedY(ys.size());

    for (std::size_t i = 0; i < data.size(); ++i) {
        sortedX[i] = data[i].first;
        sortedY[i] = data[i].second;
    }
   
    plt::plot(sortedX, sortedY);
    plt::save(filename);
    plt::close();
    plt::clf();
}

void Bayesian::Plot2D(
    const std::vector<Eigen::VectorXd>& xs,
    const std::vector<double>& ys,
    const std::string& filename) const {
    
    std::vector<double> X; 
    std::vector<double> Y;
    std::vector<double> Z;

    Z = ys;

    for (std::size_t i = 0; i != xs.size(); ++i) {
        X.push_back(xs[i][0]);
        Y.push_back(xs[i][1]);
    }
    plt::scatter(X,Y,Z);
    plt::save(filename);
    plt::close();
    plt::clf();
}
double Bayesian::SampleDev(const std::vector<double>& v, double mu) const {
    double sum_sq_diff = 0.0;
    for (double val : v) {
        double diff = val - mu;
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff / (v.size() - 1));
} 

Matrix Bayesian::ComputeCovarianceMatrix(
    const std::vector<Eigen::VectorXd>& xis,
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


double Bayesian::Kernel(
    const Eigen::VectorXd& lhs,
    const Eigen::VectorXd& rhs,
    double sigma, double lengthScale) const {
    
    double sum = (lhs - rhs).squaredNorm();  
    return sigma*sigma*std::exp(-sum / (2.0 * lengthScale * lengthScale));
}
