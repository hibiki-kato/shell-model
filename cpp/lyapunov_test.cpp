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
/**
 * @file lyapunov_test.cpp
 * @author Hibiki Kato
 * @brief lyapunov exponent test
 * @version 0.1
 * @date 2023-10-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */
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
void EigenVec2npy(Eigen::VectorXd Vec, std::string fname);
Eigen::VectorXcd npy2EigenVec<std::complex<double>>(const char* fname);
Eigen::VectorXcd perturbation(Eigen::VectorXcd state,  std::vector<int> dim, int s_min = -1, int s_max = -1);


int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    
    SMparams params;
    params.nu = 0.00018;
    params.beta = 0.417;
    params.f = std::complex<double>(1.0,0.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 2000;
    double dump =
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../../initials/beta0.417_nu0.00018_7000period_dt0.002.npy");
    ShellModel SM(params, dt, t_0, t, dump, dummy);
    int threads = omp_get_max_threads();
    int repetitions = 1000;
    std::cout << threads << "threads" << std::endl;
    std::ostringstream oss;

    ShellModel SM_origin = SM;
    std::vector<int> range(x_0.size());
    std::iota(range.begin(), range.end(), 1); // iota: 連番を作成する
    SM_origin.set_x_0_(perturbation(SM_origin.get_x_0_(), range, 0, 0)); // 初期値をランダムに与える
    ShellModel SM_another = SM;
    Eigen::VectorXcd perturbed_x_0 = perturbation(SM_origin.get_x_0_(), range, -4, -4); // create perturbed init value
    SM_another.set_x_0_(perturbed_x_0); // set above
    
    Eigen::MatrixXcd origin = SM_origin.get_trajectory_();
    Eigen::MatrixXcd another = SM_another.get_trajectory_();

    // Eigen::VectorXd delta_0 = (origin.topLeftCorner(origin.rows()-1, 1) - another.topLeftCorner(another.rows()-1, 1)).cwiseAbs();
    // Eigen::VectorXd delta_t = (origin.topRightCorner(origin.rows()-1, 1) - another.topRightCorner(another.rows()-1, 1)).cwiseAbs();
    // Eigen::VectorXd lyapunov = (delta_t.array() / delta_0.array()).log() / (SM.get_t_() - SM.get_t_0_());


    double delta_0 = (origin.topLeftCorner(origin.rows()-1, 1) - another.topLeftCorner(another.rows()-1, 1)).norm();
    double delta_t = (origin.topRightCorner(origin.rows()-1, 1) - another.topRightCorner(another.rows()-1, 1)).norm();
    double lyapunov = std::log(delta_t / delta_0) / (SM.get_t_() - SM.get_t_0_());
    

    std::cout << "largest lyapunov exponent: " << lyapunov << std::endl;
    

    myfunc::duration(start);
}