#include "Runge_Kutta.hpp"
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <math.h>
#include <random>

bool ShellModel::isLaminar_(Eigen::VectorXcd state, Eigen::MatrixXcd laminar, double epsilon)
// laminarに含まれていたらtrue, そうでなければFalseを返す
{   
    int row_start = 0;
    int row_end = state.rows() - 1;
    Eigen::VectorXd distance = (laminar.middleRows(row_start, row_end).cwiseAbs() - state.middleRows(row_start, row_end).replicate(1, laminar.cols()).cwiseAbs()).colwise().norm();

    return (distance.array() < epsilon).any();
}