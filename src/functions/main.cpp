#include "Plotter.hpp"
#include "Cone.hpp"
#include "Parabola.hpp"
#include "Rastrigin.hpp"
#include "Rosenbrock.hpp"

int main() {
    Plotter plotter;
    Cone cone;
    Parabola parabola;
    Rastrigin rastrigin;
    Rosenbrock rosenbrock;

    plotter.plot1D(cone, 100, "Cone1D.png");
    plotter.plot1D(parabola, 100, "Parabola1D.png");
    plotter.plot1D(rastrigin, 100, "Rastrigin1D.png");
    plotter.plot1D(rosenbrock, 100, "Rosenbrock1D.png");

    plotter.plot2D(cone, 100, "Cone2D.png");
    plotter.plot2D(parabola, 100, "Parabola2D.png");
    plotter.plot2D(rastrigin, 100, "Rastrigin2D.png");
    plotter.plot2D(rosenbrock, 100, "Rosenbrock2D.png");
    
    plotter.plot3D(cone, 100,20, "Cone3D.png");
    plotter.plot3D(parabola, 100,20, "Parabola3D.png");
    plotter.plot3D(rastrigin, 100,20, "Rastrigin3D.png");
    plotter.plot3D(rosenbrock, 100,20, "Rosenbrock3D.png");

    return 0;
}
