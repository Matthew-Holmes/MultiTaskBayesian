#include "Plotter.hpp"
#include "Sphere.hpp"
#include "Parabola.hpp"
#include "Rastrigin.hpp"
#include "Rosenbrock.hpp"

int main() {
    Plotter plotter;
    Sphere sphere;
    Parabola parabola;
    Rastrigin rastrigin;
    Rosenbrock rosenbrock;

    plotter.plot1D(sphere, 100, "Sphere1D.png");
    plotter.plot1D(parabola, 100, "Parabola1D.png");
    plotter.plot1D(rastrigin, 100, "Rastrigin1D.png");
    plotter.plot1D(rosenbrock, 100, "Rosenbrock1D.png");

    return 0;
}
