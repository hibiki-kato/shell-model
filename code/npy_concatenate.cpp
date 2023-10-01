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
namespace plt = matplotlibcpp;

void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname);
Eigen::MatrixXcd npy2EigenMat(const char* fname);

int main(){
    const char* a_name = "../../generated_lam/generated_laminar_beta_0.421nu_0.00018_47000period1300check200progresseps0.04.npy";
    const char* b_name = "../../generated_lam/generated_laminar_beta_0.421nu_0.00018_50000period1300check200progresseps0.04.npy";
    int check_point = 46600; // the last time of a (not equal to time in the file name, usually a nice round number)

    Eigen::MatrixXcd a = npy2EigenMat(a_name);
    Eigen::MatrixXcd b = npy2EigenMat(b_name);

    check_point *= 100;
    Eigen::MatrixXcd c(a.rows(), check_point + b.cols());
    c.leftCols(check_point) = a.leftCols(check_point);
    c.rightCols(b.cols()) = b;
    
     plt::figure_size(1200, 780);
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
    EigenMt2npy(c, b_name);

}

void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname){
    Eigen::MatrixXcd transposed = Mat.transpose();
    // map to const mats in memory
    Eigen::Map<const Eigen::MatrixXcd> MOut(&transposed(0,0), transposed.cols(), transposed.rows());
    // save to np-arrays files
    cnpy::npy_save(fname, MOut.data(), {(size_t)transposed.cols(), (size_t)transposed.rows()}, "w");
}

Eigen::MatrixXcd npy2EigenMat(const char* fname){
    std::string fname_str(fname);
    cnpy::NpyArray arr = cnpy::npy_load(fname_str);
    if (arr.word_size != sizeof(std::complex<double>)){
        throw std::runtime_error("Unsupported data type in the npy file.");
    }
    Eigen::Map<const Eigen::MatrixXcd> MatT(arr.data<std::complex<double>>(), arr.shape[1], arr.shape[0]);
    return MatT.transpose();
}