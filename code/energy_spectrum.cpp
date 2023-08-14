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
namespace plt = matplotlibcpp;

int main(){
    double nu = 0.00004;
    double beta = 0.5;
    std::complex<double> f = std::complex<double>(1.0,0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 10000;
    double latter = 10;
    Eigen::VectorXcd x_0(15);
    x_0(0) = std::complex<double>(0.4350E+00 , 0.5008E+00);
    x_0(1) = std::complex<double>(0.1259E+00 , 0.2437E+00);
    x_0(2) = std::complex<double>(-0.8312E-01 , -0.4802E-01);
    x_0(3) = std::complex<double>(0.5164E-01 , -0.1599E+00);
    x_0(4) = std::complex<double>(-0.1899E+00 , -0.3602E-01);
    x_0(5) = std::complex<double>(0.4093E-03 , 0.8506E-01);
    x_0(6) = std::complex<double>(0.9539E-01 , 0.3215E-01);
    x_0(7) = std::complex<double>(-0.5834E-01 , 0.4433E-01);
    x_0(8) = std::complex<double>(-0.8790E-02 , 0.2502E-01);
    x_0(9) = std::complex<double>(0.3385E-02 , 0.1148E-02);
    x_0(10) = std::complex<double>(-0.7072E-04 , 0.5598E-04);
    x_0(11) = std::complex<double>(-0.5238E-07 , 0.1467E-06);
    x_0(12) = std::complex<double>(0.1E-07 ,0.1E-06);
    x_0(13) = std::complex<double>(0.1E-07 ,0.1E-06);
    x_0(14) = std::complex<double>(0.1E-07 ,0.1E-06);


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