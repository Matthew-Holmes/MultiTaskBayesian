#include "OptimisationLogger.hpp"
#include <iostream>
#include <cfloat>
#include <vector>
#include <utility>
#include <random>
#include <algorithm>
#include <cmath>

std::vector<std::vector<double>> OptimisationLogger::GetEvalsFor(
        OptimiserBase& opt,
        FunctionBase& func,
        int dim,
        int maxit,
        int maxTimePerItMillisec) {
    
    LoggingWrapperFunction wrapped(func);
    
    auto [lb,ub] = GetRandomisedStartingBounds(dim);
    std::tie(lb,ub) = NormaliseSearchVolume(lb,ub,2.0);

//    for (double x : lb)
//    {
//        std::cout << x << " ";
//    }
//    std::cout << std::endl;
//    for (double x : ub)
//    {
//        std::cout << x << " ";
//    }
//    std::cout << std::endl;

    // TODO - once Bayesian done, set time parameter on it

    double bestMerit = DBL_MAX;
    std::vector<double> foundMin = opt.optimise(     
        bestMerit,
        wrapped,
        lb, ub,
        maxit);

    std::vector<std::vector<double>> ret = wrapped.evalHistory;

    ret.push_back(foundMin);
    ret.back().push_back(bestMerit);

    return ret;    
}        


double OptimisationLogger::LoggingWrapperFunction::eval(
    const std::vector<double>& input) const {

    evalHistory.push_back(input); // copies the input 
    
    double ret = baseFunction->eval(input);

    evalHistory.back().push_back(ret);
    // we'll unpack this later

    return ret;
}

OptimisationLogger::LoggingWrapperFunction::LoggingWrapperFunction(
    FunctionBase& toWrap) : baseFunction(&toWrap) {
    baseFunction =  &toWrap;    
}
    
std::pair<std::vector<double>, std::vector<double>>
OptimisationLogger::GetRandomisedStartingBounds(
    int dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> widthDist(2.0, 3.0);
    std::uniform_real_distribution<> shiftDist(-1.0, 1.0);

    std::vector<double> lb(dim);
    std::vector<double> ub(dim);

    for (int i = 0; i < dim; ++i) {
        double width = widthDist(gen);
        double shift = shiftDist(gen);

        double b1 = shift - width / 2.0;
        double b2 = shift + width / 2.0;
        
        lb[i] = std::min(b1, b2);
        ub[i] = std::max(b1, b2);
    }
    return std::make_pair(lb,ub);        
}

std::pair<std::vector<double>, std::vector<double>>
OptimisationLogger::NormaliseSearchVolume(
    std::vector<double> lb,
    std::vector<double> ub,
    double oneSideLengthEquivalent) {
    // shrinks/expands the hypercuboid spanned by the bounds lb and ub
    // resultant volume will be oneSideLengthEquivalent^dim
    // where dim is the dimensionality of the bounds

    // implementation uses log to avoid numerical issues in high dimensions

    std::cout << std::endl;
    std::size_t dim = lb.size();
    std::vector<double> logDistances(dim);
    double logDesiredVolume = log(oneSideLengthEquivalent) * dim;

    // compute the log distances spanned by the bounds
    for (int i = 0; i< dim; ++i) {
        logDistances[i] = log(ub[i] - lb[i]);
    }

    // compute log total volume
    double sumLogDistances = 0.0;
    for (double logDistance : logDistances) {
        sumLogDistances += logDistance;
    }

    // compute the difference we need to adjust
    double diff = sumLogDistances - logDesiredVolume;
    double adjustment = diff / dim;

    // adjust each bound and return from log space
    for (int i = 0; i < dim; ++i) {
        double currentLength = ub[i] - lb[i];
        double newLength = exp(log(currentLength) - adjustment);
        double midPoint = (ub[i] + lb[i]) / 2.0;
        lb[i] = midPoint - newLength / 2.0;
        ub[i] = midPoint + newLength / 2.0;
    }

    // ensure that origin, (0,0,....) remains within bounds
    for (int i = 0; i < dim; ++i) {
        if (lb[i] > 0) {
            double shift = lb[i];
            lb[i] -= shift;
            ub[i] -= shift;
        } else if (ub[i] < 0) {
            double shift = ub[i];
            lb[i] -= shift;
            ub[i] -= shift;
        }
    }
    
    return std::make_pair(lb,ub);        
}   
