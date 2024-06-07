#ifndef BAYESIAN_HPP
#define BAYESIAN_HPP

#include "OptimiserBase.hpp"
#include <vector>
#include <Eigen/Dense> // linear algebra library headers

typedef Eigen::Matrix<double, Dynamic, Dynamic> Matrix;

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
        std::vector<Eigen::VectorXd>& xis,
        Eigen::VectorXd& yis,
        int numSamples);
    
    std::vector<double> GetBestEval(
        double &bestMeritOut,
        const std::vector<Eigen::VectorXd>& xis,
        const Eigen::VectorXd& yis);        

    void DoBayesianStep(
        const FunctionBase& meritFunction,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        std::vector<Eigen::VectorXd>& xis,
        Eigen::vectorXd& yis)
    
    double SampleDev(const Eigen::VectorXd& v, double mu);    

    Matrix ComputeCovarianceMatrix(
        const std::vector<Eigen::VectorXd>& xis,
        double sigma,
        double lengthScale); 
    
    double Kernel(
        const Eigen::VectorXd& lhs,
        const Eigen::VectorXd& rhs,
        double sigma, double lengthScale);
};

#endif
