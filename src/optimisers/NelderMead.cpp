#include "NelderMead.hpp"
#include <float.h>

std::vector<double> NelderMead::optimise(
    double &bestMeritOut,
    const FunctionBase& meritFunction,
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    int maxit, int timePerItms) const {

    std::size_t dim = lb.size();

    auto nelderMeadFunction = [&](const std::vector<double> &x) {
        return meritFunction.eval(x);
    };

    std::vector<double> start(dim);
    std::vector<double> step(dim);

    for (size_t i = 0; i < dim; ++i)
    {
        start[i] = (ub[i] + lb[i]) / 2.0;
        step[i]  = (ub[i] - lb[i]) / 10;
    }

    auto result = nelder_mead<double>(
        nelderMeadFunction, start, DBL_MIN, step, INT_MAX, maxit);

    bestMeritOut = result.ynewlo;

    return result.xmin;

}

