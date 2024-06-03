#ifndef OPTIMISERBASE_HPP
#define OPTIMISERBASE_HPP

#include <vector>
#include "FunctionBase.hpp"


class OptimiserBase
{
public:
    virtual ~OptimiserBase() = default;
    virtual std::vector<double> optimise(
        int &bestMeritOut,
        const FunctionBase& meritFunction,
        const std::vector<double>& lb,
        const std::vector<double>& ub) const = 0;
};


#endif
