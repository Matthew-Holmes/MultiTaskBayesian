#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "FunctionBase.hpp"

class Sphere : public FunctionBase {
public:
    virtual double eval(const std::vector<double>& input) const override;
};

#endif
