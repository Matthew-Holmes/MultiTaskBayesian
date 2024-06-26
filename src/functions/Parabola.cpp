#include "Parabola.hpp"

double Parabola::eval(const std::vector<double>& input) const {
    double sum = 0.0;
    for (double x : input) {
        sum += x * x;
    }
    return sum;
}
