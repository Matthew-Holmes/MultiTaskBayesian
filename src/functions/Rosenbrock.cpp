#include "Rosenbrock.hpp"

double Rosenbrock::eval(const std::vector<double>& input) const {
    double sum = 0.0;

    double xi = 0.0; double xp = 0.0;
    double a  = 0.0; double b  = 0.0;

    for (size_t i = 0; i != input.size() - 1; i++) {
        xi = input[i];
        xp = input[i+1];
        
        // adjust so minimum at (0,0,..)
        // and [-1,1]^n scale maps into [-2,2]^2
        xi *= 2.0; xp *= 2.0; 
        xi += 1.0; xp += 1.0;

        a = xp - xi * xi;
        b = 1.0 - xi;
        sum += 100.0 * a * a;
        sum += b * b;
    }
    return sum;
}
