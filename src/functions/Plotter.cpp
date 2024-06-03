#include <cstring>
#include <vector>

#include "Plotter.hpp"
#include "matplotlib-cpp.h"

namespace plt = matplotlibcpp;

void Plotter::plot1D(
    const FunctionBase& func,
    const int numPoints,
    const std::string& filename) const
{
    std::vector<double> x(numPoints), y(numPoints);
    double step = 2.0 / (numPoints - 1);
    for (int i = 0; i < numPoints; ++i) {
        x[i] = -1.0 + i * step;
        y[i] = func.eval({x[i]});
    }
    plt::plot(x, y);
    plt::save(filename);
    plt::close();
}

void Plotter::plot2D(
    const FunctionBase& func,
    const int numPoints,
    const std::string& filename) const
{
    std::vector<std::vector<double>> x(
        numPoints, std::vector<double>(numPoints));
    std::vector<std::vector<double>> y(
        numPoints, std::vector<double>(numPoints));
    std::vector<std::vector<double>> z(
        numPoints, std::vector<double>(numPoints));
    double step = 2.0 / (numPoints - 1);
    for (int i = 0; i < numPoints; ++i) {
        for (int j = 0; j < numPoints; ++j) {
            x[i][j] = -1.0 + i * step;
            y[i][j] = -1.0 + j * step;
            z[i][j] = func.eval({x[i][j], y[i][j]});
        }
    }
    plt::contour(x,y,z);
    plt::save(filename);
    plt::close();
}

void Plotter::plot3D(
    const FunctionBase& func,
    const int numPoints,
    const int numSlices,
    const std::string& filenamePrefix) const
{
    double step = 2.0 / (numPoints - 1);
    double sliceStep = 2.0 / (numSlices - 1);
    for (int i = 0; i < numSlices; ++i) {
        std::vector<std::vector<double>> x(
            numPoints, std::vector<double>(numPoints));
        std::vector<std::vector<double>> y(
            numPoints, std::vector<double>(numPoints));
        std::vector<std::vector<double>> z(
            numPoints, std::vector<double>(numPoints));
        double slice = -1.0 + i * sliceStep;
        for (int j = 0; j < numPoints; ++j) {
            for (int k = 0; k < numPoints; ++k) {
                x[j][k] = -1.0 + j * step;
                y[j][k] = -1.0 + k * step;
                z[j][k] = func.eval({slice, x[j][k], y[j][k]});
            }
        }
        std::string filename = filenamePrefix + "_" + std::to_string(i) + ".png";
        plt::contour(x,y,z); 
        plt::save(filename);
        plt::clf(); // Clear figure for next iteration
    }
}
