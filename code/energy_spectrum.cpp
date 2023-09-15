#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include "Runge_Kutta.hpp"
#include <chrono>
#include <random>
#include <iomanip>
#include "matplotlibcpp.h"
#include "cnpy/cnpy.h"

namespace plt = matplotlibcpp;
Eigen::VectorXcd npy2EigenVec(const char* fname);

int main(){
    double nu = 0.00004;
    double beta = 0.5;
    std::complex<double> f = std::complex<double>(1.0,0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 10000;
    double latter = 10;
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.5_nu1e-05_15dim_period.npy");


    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間
    ShellModel solver(nu, beta, f, ddt, t_0, t, latter, x_0);

    Eigen::MatrixXd sum_ = solver.get_energy_spectrum_();

    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end-start).count(); //処理に要した時間をミリ秒に変換
    std::cout << elapsed << std::endl;

    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "20";
    plt::rcparams(plotSettings);
    plt::figure_size(1000, 1000);
    Eigen::VectorXd k_n_ = solver.get_k_n_();
    std::vector<double> k_n(x_0.size());
    for (int i = 0; i < x_0.size(); i++) {
        k_n[i] = k_n_(i);
    }

    std::vector<double> sum(sum_.size());
    for (int i = 0; i < sum_.size(); i++) {
        sum[i] = sum_(i);
    }
    plt::scatter(k_n, sum, 10);
    plt::xlabel("$k_n$");
    plt::ylabel("$E(k_n)$");
    plt::xscale("log");
    plt::yscale("log");
    plt::save("../../energy.png");
}

Eigen::VectorXcd npy2EigenVec(const char* fname){
    std::string fname_str(fname);
    cnpy::NpyArray arr = cnpy::npy_load(fname_str);
    if (arr.word_size != sizeof(std::complex<double>)) {
        throw std::runtime_error("Unsupported data type in the npy file.");
    }
    std::complex<double>* data = arr.data<std::complex<double>>();
    Eigen::Map<Eigen::VectorXcd> vec(data, arr.shape[0]);
    return vec;
}