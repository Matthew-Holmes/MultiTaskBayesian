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

    plotter.plot2D(sphere, 100, "Sphere2D.png");
    plotter.plot2D(parabola, 100, "Parabola2D.png");
    plotter.plot2D(rastrigin, 100, "Rastrigin2D.png");
    plotter.plot2D(rosenbrock, 100, "Rosenbrock2D.png");
    
    plotter.plot3D(sphere, 100, 5, "Sphere3D.png");
    plotter.plot3D(parabola, 100, 5, "Parabola3D.png");
    plotter.plot3D(rastrigin, 100, 5, "Rastrigin3D.png");
    plotter.plot3D(rosenbrock, 100, 5, "Rosenbrock3D.png");

    return 0;
}
