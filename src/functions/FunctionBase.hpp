#ifndef FUNCTIONBASE_HPP
#define FUNCTIONBASE_HPP

#include <vector>

class FunctionBase {
public:
    virtual ~FunctionBase() = default;
    virtual double eval(const std::vector<double>& input) const = 0;
};
#endif
