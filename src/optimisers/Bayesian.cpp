#include "Bayesian.hpp"
#include <random>
#include <vector>

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

    std::vector<std::vector<double>> xis;
    std::vector<double> yis; 

    // **************************  burn in ************************************

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
    std::vector<std::vector<double>>& xis,
    std::vector<double>& yis,
    int numSamples) {
    
    std::size_t dim = lb.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); 
    
    for (int i = 0; i < numSamples; ++i) {
        std::vector<double> x(dim);
        for (std::size_t j = 0; j < dim; ++j) {
            x[j] = lb[j] + dis(gen) * (ub[j] - lb[j]);
        }
        double y = meritFunction(x);
        xis.push_back(x);
        yis.push_back(y);
    }
}


std::vector<double> GetBestEval(
    double &bestMeritOut,
    const std::vector<std::vector<double>>& xis,
    const std::vector<double>& yis) {

    auto it = std::min_element(yis.begin(), yis.end());
    std::size_t bestIndex = std::distance(yis.begin(),it);
    bestMeritOut = yis[bestIndex];
    return xis[bestIndex];
}


