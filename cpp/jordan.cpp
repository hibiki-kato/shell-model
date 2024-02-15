/**
 * @file jordan.cpp
 * @author hibiki kato
 * @brief 格子を用意して，軌道の点を中に含む格子の個数で軌道を特徴付ける
 * @version 0.1
 * @date 2024-01-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <chrono>
#include "shared/Flow.hpp"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/myFunc.hpp"
#include "shared/matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 1.8e-4;
    params.beta = 0.417;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 1e-2;
    double t_0 = 0;
    double t = 5e+4;
    double dump = 1e+5;
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../initials/beta0.417_nu0.00018_13348period_dt0.01eps0.005.npy", true);
    // Eigen::VectorXd x_0 = npy2EigenVec<double>("../initials/chaotic.npy", true);
    std::vector<std::tuple<int, double, double>> intervals;
    intervals.push_back(std::make_tuple(3, 0.05, 0.4 ));
    intervals.push_back(std::make_tuple(4, 0, 0.35));
    int mesh_num = 100;
    ShellModel SM(params, dt, t_0, t, dump, x_0);

    // calculating
    // Eigen::MatrixXd trajectory = CR.get_trajectory().cwiseAbs();
    // std::string suffix = "burst";
    // loading
    Eigen::MatrixXd trajectory = npy2EigenMat<std::complex<double>>("../sync/npy/sync_beta0.417nu0.00018_1e+06period100000window_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy", true).cwiseAbs();
    std::string suffix = "sync";

    int num = myfunc::count_mesh(trajectory, mesh_num, intervals);
    std::cout << num << std::endl;

    // intervalsを保存する
    std::ostringstream oss;
    oss << "../../jordan/" << suffix << "_beta" << params.beta << "nu" << params.nu << "t" << static_cast<int>(t-t_0) << "num" << num << "mesh" << mesh_num << ".txt";
    std::string filename = oss.str();
    std::ofstream ofs(filename);
    for (int i = 0; i < intervals.size(); ++i) {
        ofs << std::get<0>(intervals[i]) << "\t" << std::get<1>(intervals[i]) << "\t" << std::get<2>(intervals[i]) << std::endl;
    }
    ofs.close();
    myfunc::duration(start);
}