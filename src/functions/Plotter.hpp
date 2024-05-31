#ifndef PLOTTER_HPP
#define PLOTTER_HPP

#include <string>
#include "FunctionBase.hpp"

class Plotter {
public:
    void plot1D(
        const FunctionBase& func,
        const int numPoints,
        const std::string& filename) const;
    void plot2D(
        const FunctionBase& func,
        const int numPoints,
        const std::string& filename) const;
    void plot3D(
        const FunctionBase& func,
        const int numPoints,
        const int numSlices,
        const std::string& filenamePrefix) const;
};

#endif
