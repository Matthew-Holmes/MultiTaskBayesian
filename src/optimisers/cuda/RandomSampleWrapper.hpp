#ifndef RANDOMSAMPLEWRAPPER_HPP
#define RANDOMSAMPLEWRAPPER_HPP

#include <vector>
#include <Eigen/Dense>
#include <utility>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;

std::pair<std::vector<double>, double> GetBestRandomSample(
    const Matrix& K,
    const std::vector<Eigen::VectorXd> S,
    double sg, double l, /* kernel params */
    std::vector<double>& yDiff,
    double a, /* explore vs exploit */
    const std::vector<double>& lb,
    const std::vector<double>& ub,
    const int seed);


#endif
