#ifndef ROSENBROCK_HPP 
#define ROSENBROCK_HPP

#include "FunctionBase.hpp"

class Rosenbrock : public FunctionBase {
public:
    virtual double eval(const std::vector<double>& input) const = 0;
};

#endif
