#ifndef RASTRIGIN_HPP
#define RASTRIGIN_HPP

#include "FunctionBase.hpp"

class Rastrigin : public FunctionBase {
public:
    virtual double eval(const std::vector<double>& input) const = 0;
};

#endif
