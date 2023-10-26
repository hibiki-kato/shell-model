/**
 * @file npy_concatenate.cpp
 * @author Hibiki Kato
 * @brief 2 npy files concatenation. This code basically used for concatenating generated laminar.
 * @version 0.1
 * @date 2023-06-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <complex>
#include "matplotlibcpp.h"
#include "cnpy/cnpy.h"
#include "Eigen_numpy_converter.hpp"

namespace plt = matplotlibcpp;

int main(){
    const char* a_name = "../../generated_lam/sync_gen_laminar_beta_0.423nu_0.00018_dt0.01_50000period1000check100progress10^-7-10^-3perturb_4-7_4-10_4-13_7-10_7-13_10-13_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy";
    const char* b_name = "../../generated_lam/sync_gen_laminar_beta_0.423nu_0.00018_dt0.01_100000period1000check100progress10^-7-10^-4perturb_4-7_4-10_4-13_7-10_7-13_10-13_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy";
    int check_point = 50000; // the last time of a (not equal to time in the file name, usually a nice round number)

    Eigen::MatrixXcd a = npy2EigenMat<std::complex<double>>(a_name);
    Eigen::MatrixXcd b = npy2EigenMat<std::complex<double>>(b_name);

    check_point *= 100;
    Eigen::MatrixXcd c(a.rows(), check_point + b.cols());
    c.leftCols(check_point) = a.leftCols(check_point);
    c.rightCols(b.cols()) = b;
    
     plt::figure_size(780, 780);
    // Add graph titlecc
    std::vector<double> x(c.cols()),y(c.cols());
    for(int i=0;i<c.cols();i++){
        x[i]=c.cwiseAbs()(14, i);
        y[i]=i;
    }

    plt::plot(x,y);
    plt::save("../../test_concatenate.png");
    std::cout << "Succeed?" << std::endl;
    char none;
    std::cin >> none;
    EigenMat2npy(c, b_name);

}