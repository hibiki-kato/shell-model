#include <eigen3/Eigen/Dense>
#include <iostream>
#include <complex>
#include "matplotlibcpp.h"
#include "cnpy/cnpy.h"
namespace plt = matplotlibcpp;

void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname);
Eigen::MatrixXcd npy2EigenMat(const char* fname);

int main(){
    Eigen::MatrixXcd a = npy2EigenMat("../generated_laminar_beta_0.421nu_0.00018_58400period1300check400progresseps0.1.npy");
    Eigen::MatrixXcd b = npy2EigenMat("../generated_laminar_beta_0.421nu_0.00018_72400period1300check400progresseps0.1.npy");
    
    int check_point = 5600000;
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
    plt::save("../test.png");

    EigenMt2npy(c, "../generated_laminar_beta_0.421nu_0.00018_72400period1300check400progresseps0.1.npy");

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