                                                      █                                     █
█████                            █                    █     ██                              █         ██
█    █                                                █     ██                              █         ██
█    █   ███   █████ ███   ███   █  █ ███   ███    ████    ████  ███      █   █  █████   ████   ███  ████  ███
█    █  ██  █  ██  ██  █  █  ██  █  ██  █  ██  █  ██  █     ██  ██  █     █   █  ██  █  ██  █  █  ██  ██  ██  █
█████   █   █  █   █   ██     █  █  █   █  █   █  █   █     ██  █   ██    █   █  █   ██ █   █      █  ██  █   █
█   █   █████  █   █   ██  ████  █  █   █  █████  █   █     ██  █    █    █   █  █   ██ █   █   ████  ██  █████
█   ██  █      █   █   ██ █   █  █  █   █  █      █   █     ██  █   ██    █   █  █   ██ █   █  █   █  ██  █
█    █  ██  █  █   █   ██ █  ██  █  █   █  ██  █  ██  █     ██  ██  █     █   █  ██  █  ██  █  █  ██  ██  ██  █
█    ██  ████  █   █   ██ █████  █  █   █   ████   ████      ██  ███       ████  █████   ████  █████   ██  ████
                                                                                 █
                                                                                 █
                                                                                 █

#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include <chrono>
#include <random>
#include "cnpy/cnpy.h"
#include "shared/matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    
    SMparams params;
    params.nu = 0.00018;
    params.beta = 0.41525 ;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt =.01;
    double t_0 = 0;
    double t = 100000;
    double dump = 1;
    int dim = 14;

    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../../initials/beta0.415_nu0.00018_100000period_dt0.01.npy");
    std::vector<std::complex<double>> x_0_vec(x_0.data(), x_0.data() + x_0.size());
    for(int i = 0; i < dim - x_0.size(); i++){
        x_0_vec.push_back(std::complex<double>(1e-6, 1e-6));
    }
    x_0 = Eigen::Map<Eigen::VectorXcd>(x_0_vec.data(), x_0_vec.size());
    std::cout << x_0 << std::endl;
    ShellModel SM(params, dt, t_0, t, dump, dummy);
    std::ostringstream oss;
    Eigen::VectorXcd result = SM.get_trajectory_().topRightCorner(dim, 1);

    std::cout << result << std::endl;
    oss << "../../initials/beta" << beta << "_nu" << nu<< "_" << dim << "dim_period.npy";  // 文字列を結合する
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << fname << std::endl;
    EigenVec2npy(result, fname);

    myfunc::duration(start);
}