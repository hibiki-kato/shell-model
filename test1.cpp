#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include "Runge_Kutta.hpp"
#include <chrono>
#include <random>
#include <omp.h>
#include "matplotlibcpp.h"
#include "cnpy/cnpy.h"
namespace plt = matplotlibcpp;
Eigen::VectorXcd npy2EigenVec(const char* fname);
void EigenMt2npy(Eigen::MatrixXd Mat, std::string fname);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    
    double nu = 0.00017520319481270297;
    double beta = 0.416;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 1000;
    double latter = 5;
    Eigen::VectorXcd x_0 = npy2EigenVec("../beta0.416_nu0.00017520319481270297_step0.01_10000.0period_laminar.npy");
    ShellModel SM(nu, beta, f, ddt, t_0, t, latter, x_0);
    Eigen::MatrixXcd laminar = SM.get_trajectory_();
    int numRows = laminar.cols() / 10;
    Eigen::MatrixXcd laminar_sample(laminar.rows(), numRows);
    for (int i = 0; i < numRows; i++){
        int colIdx = 10 * i;
        laminar_sample.col(i) = laminar.col(colIdx);
    }
    plt::figure_size(1200, 780);
    std::vector<double> x(laminar_sample.cols()),y(laminar_sample.cols());

    for(int i=0;i<laminar_sample.cols();i++){
        x[i]=laminar_sample.cwiseAbs()(14, i);
        y[i]=laminar_sample.cwiseAbs()(0, i);
    }
    plt::plot(x,y);
    plt::save("laminar_model.png");
    t=10000;
    latter = 10;
    int skip = 10;
    double epsilon = 1;
    int threads = omp_get_max_threads();
    std::cout << threads << "threads" << std::endl;

    LongLaminar LL(nu, beta, f, ddt, t_0, t, latter, x_0, laminar_sample, epsilon, skip, 10, 10, threads);
    
    
    std::cout << SM.get_trajectory_() <<std::endl;