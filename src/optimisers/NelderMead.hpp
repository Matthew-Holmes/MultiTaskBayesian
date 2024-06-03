#ifndef NELDERMEAD_HPP
#define NELDERMEAD_HPP

#include "OptimiserBase.hpp"
#include "nelder_mead.h" // lib header

class NelderMead : public OptimiserBase {
public:
    virtual std::vector<double> optimise(
        double &bestMeritOut,
        const FunctionBase& meritFunction,
        const std::vector<double>& lb,
        const std::vector<double>& ub,
        int maxit) const override;
};

#endif
