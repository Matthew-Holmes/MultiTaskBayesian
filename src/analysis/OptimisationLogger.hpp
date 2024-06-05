#ifndef OPTIMISATIONLOGGER_HPP
#define OPTIMISATIONLOGGER_HPP

#include <vector>
#include <utility>
#include "FunctionBase.hpp"
#include "OptimiserBase.hpp"

class OptimisationLogger
{
public:
     std::vector<std::vector<double>> GetEvalsFor(
        OptimiserBase& opt,
        FunctionBase& func,
        int dim,
        int maxit,
        int maxTimePerItMillisec);

private:
    class LoggingWrapperFunction : public FunctionBase
    {
    public:
        std::vector<std::vector<double>> evalHistory;
        double eval(const std::vector<double>& input) const override;        
        LoggingWrapperFunction(FunctionBase& toWrap);
    private:
        FunctionBase* baseFunction;        
    };

    std::pair<std::vector<double>, std::vector<double>>
    GetRandomisedStartringBounds(int dim);

    std::pair<std::vector<double>, std::vector<double>>
    NormaliseSearchVolume(
        std::vector<double> lb,
        std::vector<double> ub,
        double oneSideLengthEquivalent);
};


#endif
