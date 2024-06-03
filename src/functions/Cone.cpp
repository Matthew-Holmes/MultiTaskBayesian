#include "Cone.hpp"
#include "cmath"

double Cone::eval(const std::vector<double>& input) const {
    double sum = 0.0;
    for (double x : input) {
        sum += x * x;
    }
    return std::sqrt(sum);
}
