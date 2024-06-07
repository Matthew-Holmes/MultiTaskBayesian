#include "Bayesian.hpp"
#include <random>
#include <vector>
#include <algorithm>

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
    Eigen::VectorXd yis; 

    // burn in for longer when in higher dimensions
    // need to avoid a degenerate burn in geometry
    int burnIn = 8 + dim;

    if (maxit <= burnIn) {
        // low maxit edge case
        DoBurnIn(meritFunction, lb, ub, xis, yis, maxit);
        return GetBestEval(bestMeritOut, xis, yis); 
        
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
    std::vector<Eigen::VectorXd>& xis;
    Eigen::VectorXd& yis,
    int numSamples) {
    
    std::size_t dim = lb.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); 
    
    xis.reserve(numSamples);
    yis.resize(numSamples);    

    for (int i = 0; i < numSamples; ++i) {
        Eigen::VectorXd x(dim);
        for (std::size_t j = 0; j < dim; ++j) {
            x[j] = lb[j] + dis(gen) * (ub[j] - lb[j]);
        }
        double y = meritFunction(
            std::vector<double>(x.data(), x.data() + x.size());
        xis.push_back(x);
        yis[i] = y;
    }
}


std::vector<double> Bayesian::GetBestEval(
    double &bestMeritOut,
    const std::vector<Eigen::VectorXd>& xis,
    const Eigen::VectorXd& yis) {

    Eigen::Index bestIndex;
    bestMeritOut = yis.minCoeff(&bestIndex);
    Eigen::VectorXd bestVector = xis[bestIndex];
    return std::vector<double>(
        bestVector.data(), bestVector.data() + bestVector.size());
}

void Bayesian::DoBayesianStep(
    const FunctionBase& meritFunction,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    std::vector<Eigen::VectorXd>& xis,
    Eigen::VectorXd& yis) {

    // Notation: average of recorded values - mu
    // Notation: sqrt(variance of recorded) - sigma (sg)

    double mu = yis.mean();
    double sg = SampleDev(yis, mu);    

    // we'll automate length scale finding later, now use 0.4 as default
    Matrix cov = ComputeCovarianceMatrix(xis, sg, 0.4);
    Matrix K = cov.inverse();

    // produce a lambda surrogate function for mu_pred, sg_pred
    auto surrogate = [&] (const Eigen::VectorXd xp) {

        Eigen::VectorXd   dists(xis.size());
        Eigen::VectroXd weights(xis.size());

        for (int i = 0; i < xis.size(); ++i) {
            dists[i] = Kernel(xp, xis[i], sg, 0.4);}

        weights = K * dists;
        
        double mu_pred = weights.dot(yis);
        double sg_pred = sg - weights.dot(dists);
        
        return std::make_pair(mu_pred, sg_pred);
    }
        
    // optimise that (random sample)

    // eval meritFunction

    // update xis, yis
}
        

double Bayesian::SampleDev(const Eigen::VectorXd& v, double mu) {
    double sum_sq_diff = (v.array() - mean).square().sum();
    return sqrt(sum_sq_diff / (v.size() - 1)); // -1 since sample stdDev
} 

Matrix Bayesian::ComputeCovarianceMatrix(
    const std::vector<Eigen::VectorXd>& xis,
    double sigma,
    double lengthScale) {

    size_t n = xis.size();
    Matrix covarianceMatrix(n,n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            covarianceMatrix(i,j) = Kernel(
                xis[i], xis[j], sigma, lengthScalejk);   
        }
    }

    return covarianceMatrix;
} 


double Bayesian::Kernel(
    const Eigen::VectorXd& lhs,
    const Eigen::VectorXd& rhs,
    double sigma, double lengthScale) {
    
    double sum = (lhs - rhs).squaredNorm();  
    return sigma*sigma*std::exp(-sum / (2.0 * lengthScale * lengthScale));
}
