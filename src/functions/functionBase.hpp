#ifndef FUNCTIONBASE_H
#define FUNCTIONBASE_H

#include <vector>

class functionBase {
public:
    virtual ~functionBase() = default;
    virtual double eval(std::vector<double>& input) const = 0;
};
#endif
