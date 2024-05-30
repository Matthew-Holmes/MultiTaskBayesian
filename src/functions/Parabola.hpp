#ifndef PARABOLA_HPP
#define PARABOLA_HPP

#include "FunctionBase.hpp"

class Parabola : public FunctionBase {
public:
    virtual double eval(const std::vector<double>& input) const override;
};

#endif
