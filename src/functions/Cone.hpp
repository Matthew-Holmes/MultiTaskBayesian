#ifndef CONE_HPP
#define CONE_HPP

#include "FunctionBase.hpp"

class Cone : public FunctionBase {
public:
    virtual double eval(const std::vector<double>& input) const override;
};

#endif
