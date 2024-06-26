#ifndef BAYESIAN_HPP
#define BAYESIAN_HPP

#include "OptimiserBase.hpp"
#include "OptimisationPolicy.hpp"
#include <vector>
#include <Eigen/Dense> // linear algebra library headers

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
//typedef Eigen::MatrixXd Matrix;

class Bayesian : public OptimiserBase {
public:
    virtual std::vector<double> optimise(
        double &bestMeritOut,
        const FunctionBase& meritFunction,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        int maxit, int timePerItms) const override;
   
private: 
    void DoBurnIn(
        const FunctionBase& meritFunction,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        std::vector<Eigen::VectorXd>& xis,
        std::vector<double>& yis,
        int numSamples) const;
    
    std::vector<double> GetBestEval(
        double &bestMeritOut,
        const std::vector<Eigen::VectorXd>& xis,
        const std::vector<double>& yis) const;        

    void DoBayesianStep(
        const FunctionBase& meritFunction,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        std::vector<Eigen::VectorXd>& xis,
        std::vector<double>& yis,
        const float mu,
        const float sg,
        OptimisationPolicy& policy) const;
    
    std::vector<bool> GenerateRandomMask(int parentSize, int samples) const;
 
    void Plot1D(
        const std::vector<Eigen::VectorXd>& xs,
        const std::vector<double>& ys,
        const std::string& filename) const;

    void Plot2D(
        const std::vector<Eigen::VectorXd>& xs,
        const std::vector<double>& ys,
        const std::string& filename) const;
        
    double SampleDev(const std::vector<double>& v, double mu) const;    

    Matrix ComputeCovarianceMatrix(
        const std::vector<Eigen::VectorXd>& xis,
        double sigma,
        double lengthScale) const; 
    
    double Kernel(
        const Eigen::VectorXd& lhs,
        const Eigen::VectorXd& rhs,
        double sigma, double lengthScale) const;
};

#endif
