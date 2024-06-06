#ifndef BAYESIAN_HPP
#define BAYESIAN_HPP

#include "OptimiserBase.hpp"
#include <vector>
#include <Eigen/Dense> // linear algebra library headers


class Bayesian : public OptimiserBase {
public:
    virtual std::vector<double> optimise(
        double &bestMeritOut,
        const FunctionBase& meritFunction,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        int maxit) const override;
   
private: 
    void DoBurnIn(
        const FunctionBase& meritFunction,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        std::vector<std::vector<double>>& xis,
        std::vector<double>& yis,
        int numSamples);
    
    std::vector<double> GetBestEval(
        double &bestMeritOut,
        const std::vector<std::vector<double>>& xis,
        const std::vector<double>& yis);        

};

#endif
