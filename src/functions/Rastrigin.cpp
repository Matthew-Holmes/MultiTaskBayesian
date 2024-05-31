#include "Rastrigin.hpp"
#include <cmath>
#include <math.h>

double Rastrigin::eval(const std::vector<double>& input) const {
    double sum = 10.0 * input.size();
    for (double x : input) {
        x *= 5; // so [-1,1] maps to [-5,5] range of Rastrigin
        sum += x * x;
        sum -= 10.0 * std::cos(2.0 * M_PI * x);    
    }
    return sum;
}
