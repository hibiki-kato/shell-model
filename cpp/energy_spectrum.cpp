#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
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
    double t = 1e+4;
    double dump = 1e+6;
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../initials/beta0.5_nu4e-05_15dim.npy", true);

    ShellModel SM(params, dt, t_0, t, dump, x_0);
    // Eigen::MatrixXcd trajectory = npy2EigenMat("../../generated_lam/generated_laminar_beta_0.417nu_0.00018_dt0.01_50000period1300check200progresseps0.05.npy");

    Eigen::MatrixXd sum_ = SM.energy_spectrum();

    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "25";
    plt::rcparams(plotSettings);
    plt::figure_size(1000, 1000);
    std::vector<double> k_n(SM.k_n.data(), SM.k_n.data() + SM.k_n.size());
    std::vector<double> sum(sum_.data(), sum_.data() + sum_.size());
    plt::scatter(k_n, sum, 10);
    plt::xlabel("$k_n$");
    plt::ylabel("$E(k_n)$");
    plt::xscale("log");
    plt::yscale("log");

    std::stringstream oss;
    oss << "../../energy/beta" << params.beta << "nu" << params.nu << "dt" << dt << "t" << t << "dim" << x_0.size() << ".png";
    std::string plotfname = oss.str();
    std::cout << "saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    myfunc::duration(start);
}