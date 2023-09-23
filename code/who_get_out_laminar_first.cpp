/**
 * @file who_get_out_laminar_first.cpp
 * @author Hibiki Kato
 * @brief observe trajectory and detect which shell get out laminar first
 * @version 0.1
 * @date 2023-09-23
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
#include "Runge_Kutta.hpp"
#include <chrono>
#include <random>
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname);
Eigen::VectorXcd npy2EigenVec(const char* fname);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double nu = 0.00018;
    double beta = 0.419;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 2e+4;
    double latter = 2;
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.41616nu0.00018_1.00923e+06period.npy");
    ShellModel solver(nu, beta, f, ddt, t_0, t, latter, x_0);
    Eigen::MatrixXcd trajectory = solver.get_trajectory_(); 
    int numCols = trajectory.cols() / 500;
    Eigen::MatrixXcd laminar_sample(trajectory.rows(), numCols);
    for (int i = 0; i < numCols; i++){
        int colIdx = 500 * i;
        laminar_sample.col(i) = trajectory.col(colIdx);
    }
    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1200, 1200);
    // Add graph title
    std::vector<double> x(laminar_sample.cols()),y(laminar_sample.cols());
    
    oss.str("");
     // 文字列を取得する
    oss << "../../beta" << beta << "_nu" << nu <<"_"<< t-t_0 << "period.npy";  // 文字列を結合する
    std::string npyfname = oss.str();
    // std::cout << "Saving result to " << npyfname << std::endl;
    // EigenMt2npy(laminar_sample, npyfname);

    auto end = std::chrono::system_clock::now();  // 計測終了時間
    int hours = std::chrono::duration_cast<std::chrono::hours>(end-start).count(); //処理に要した時間を変換
    int minutes = std::chrono::duration_cast<std::chrono::minutes>(end-start).count(); //処理に要した時間を変換
    int seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count(); //処理に要した時間を変換
    int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間を変換
    std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;
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

void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname){
    Eigen::MatrixXcd transposed = Mat.transpose();
    // map to const mats in memory
    Eigen::Map<const Eigen::MatrixXcd> MOut(&transposed(0,0), transposed.cols(), transposed.rows());
    // save to np-arrays files
    cnpy::npy_save(fname, MOut.data(), {(size_t)transposed.cols(), (size_t)transposed.rows()}, "w");
}

int isLaminar(Eigen::VectorXcd x){
    if (x.rows() != 14){
        throw std::runtime_error("x is not 14 dimentional");
    }
    if((0.2 > x(0).real() | x(0).real() > 0.6)
        | (0.4 > x(0).imag() | x(0).imag() > 0.75)

        | (x(1).cwiseAbs() > 0.5)
    
        | (x(2).cwiseAbs() > 0.6)
    
        | (-0.1 > x(3).real() | x(3).real() > 0.3)
        | (0 > x(3).imag() | x(3).imag() > 0.3)
    
        | (x(4).cwiseAbs() > 0.3)
    
        | (x(5).cwiseAbs() > 0.25)
    
        | (-0.15 > x(6).real() | x(6).real() > 0.25)
        | (0 > x(6).imag() | x(6).imag() > 0.25)){
    
        return false;
    }
    x(0)
}

