#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <chrono>
#include <random>
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/matplotlibcpp.h"
namespace plt = matplotlibcpp;
 
int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 4e-5;
    params.beta = 0.5;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+5;
    double dump = 0;
    int dim = 15;

    Eigen::VectorXcd x_0 = Eigen::VectorXcd::Random(dim) * 1e-5;
    ShellModel SM(params, dt, t_0, t, dump, x_0);
    for (int i = 0; i < SM.steps; i++) {
        SM.x_0 = SM.rk4(SM.x_0);
    }
    Eigen::VectorXcd result = SM.x_0;

    std::cout << result << std::endl;
    std::ostringstream oss;
    oss << "../../initials/beta" << params.beta << "_nu" << params.nu<< "_" << dim << "dim.npy";  // 文字列を結合する
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << fname << std::endl;
    EigenVec2npy(result, fname);

    myfunc::duration(start);
}