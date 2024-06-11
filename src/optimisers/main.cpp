#include "Bayesian.hpp"
#include "NelderMead.hpp"
#include "Rastrigin.hpp"
#include "Parabola.hpp"
#include "Rosenbrock.hpp"

#include <iostream>
#include <float.h>

int main() {
    NelderMead nm;
    Bayesian bay;
    Parabola parabola;
    Rastrigin rastrigin;
    Rosenbrock rosenbrock;
    
    double bestMerit = DBL_MAX;
    std::vector<double> lb = {-1.412, -1.323};//, -1.412, -1.412};
    std::vector<double> ub = {0.598, 0.677};//, 0.412, 0.323};

    std::vector<double> ret;

    ret = bay.optimise(bestMerit,rosenbrock, lb, ub, 50, 1000);

    std::cout << bestMerit << std::endl;
    
    for (double x : ret)
    {
        std::cout << x << " ";
    }

    std::cout << std::endl;

    return 0;

}
