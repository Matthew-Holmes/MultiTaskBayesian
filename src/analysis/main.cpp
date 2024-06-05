#include "OptimisationLogger.hpp"

#include "Rosenbrock.hpp"
#include "NelderMead.hpp"
#include "Cone.hpp"

#include <iostream>
#include <float.h>

int main() {
    NelderMead nm;
    Rosenbrock rosenbrock;
    Cone cone;

    OptimisationLogger logger;

    std::vector<std::vector<double>> logs = logger.GetEvalsFor(
        nm,
        rosenbrock,
        5,
        1000,
        1000);
    
    for (std::vector<double> vec : logs)
    {
        for (double x : vec)
        {
            std::cout << x << " ";
        }
        std::cout << std::endl; 
    }        

    return 0;
}

