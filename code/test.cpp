#include <eigen3/Eigen/Dense>
#include <complex>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <type_traits>
// #include <cmath>
#include "matplotlibcpp.h"
#include "cnpy/cnpy.h"
namespace plt = matplotlibcpp;
Eigen::MatrixXcd npy2EigenMat(const char* fname);
Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& state, Eigen::VectorXd k_n, double beta, double nu);
Eigen::MatrixXd readMatrixFromFile(const std::string& filename);
Eigen::MatrixXcd npy2EigenMatR(const char* fname);

int main(){
    double nu = 0.1;
    double beta = 0.4;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 10000;
    double latter = 1;
    Eigen::VectorXcd x_0(6);
    x_0(0) = std::complex<double>(1 ,1);
    x_0(1) = std::complex<double>(1 ,1);
    x_0(2) = std::complex<double>(1 ,1);
    x_0(3) = std::complex<double>(1 ,1);
    x_0(4) = std::complex<double>(1 ,1);
    x_0(5) = std::complex<double>(1 ,1);

    Eigen::VectorXd k_n(x_0.rows());
    for (int i = 0; i < x_0.rows(); ++i) {
        k_n(i) = pow(2, i-3);
        x_0(i) = std::complex<double>(1, 1);
    }
    
    Eigen::VectorXd state(x_0.rows()*2);
    for (int i = 0; i < x_0.rows(); ++i) {
        state(2*i) = x_0(i).real();
        state(2*i+1) = x_0(i).imag();
    }
    computeJacobian(state, k_n, beta, nu);
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

// ヤコビアンの計算
Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& state, Eigen::VectorXd k_n, double beta, double nu){
    int dim = state.rows();
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(dim, dim);
    
    // A
    for (int i = 0; i < dim/2 - 2; ++i) {
        jacobian(2*i, 2*i + 2) += k_n(i) * state((i+2)*2 + 1);
        jacobian(2*i, 2*i+1 + 2) += k_n(i) * state((i+2)*2);
        jacobian(2*i+1, 2*i + 2) += k_n(i) * state((i+2)*2);
        jacobian(2*i+1, 2*i+1 + 2) += -k_n(i) * state((i+2)*2 + 1);

        jacobian(2*i, 2*i + 4) +=  k_n(i) * state((i+1)*2 + 1);
        jacobian(2*i, 2*i+1 + 4) += k_n(i) * state((i+1)*2);
        jacobian(2*i+1, 2*i + 4) +=  k_n(i) * state((i+1)*2);
        jacobian(2*i+1, 2*i+1 + 4) += -k_n(i) * state((i+1)*2 + 1);
    }

    // B
    for (int i = 1; i < dim/2 - 1; ++i) {
        jacobian(2*i, 2*i - 2) +=  -beta * k_n(i-1) * state((i+1)*2 + 1);
        jacobian(2*i, 2*i+1 - 2) += -beta * k_n(i-1) * state((i+1)*2);
        jacobian(2*i+1, 2*i - 2) += -beta * k_n(i-1) * state((i+1)*2);
        jacobian(2*i+1, 2*i+1 - 2) += beta * k_n(i-1) * state((i+1)*2 + 1);

        jacobian(2*i, 2*i + 2) +=  -beta * k_n(i-1) * state((i-1)*2 + 1);
        jacobian(2*i, 2*i+1 + 2) += -beta * k_n(i-1) * state((i-1)*2);
        jacobian(2*i+1, 2*i + 2) +=  -beta * k_n(i-1) * state((i-1)*2);
        jacobian(2*i+1, 2*i+1 + 2) += beta * k_n(i-1) * state((i-1)*2 + 1);
    }

    // Gamma
    for (int i = 2; i < dim/2; ++i) {
        jacobian(2*i, 2*i - 4) +=  (beta-1) * k_n(i-2) * state((i-1)*2 + 1);
        jacobian(2*i, 2*i+1 - 4) += (beta-1) * k_n(i-2) * state((i-1)*2);
        jacobian(2*i+1, 2*i - 4) += (beta-1) * k_n(i-2) * state((i-1)*2);
        jacobian(2*i+1, 2*i+1 - 4) += (1-beta) * k_n(i-2) * state((i-1)*2 + 1);

        jacobian(2*i, 2*i - 2) +=  (beta-1) * k_n(i-2) * state((i-2)*2 + 1);
        jacobian(2*i, 2*i+1 - 2) += (beta-1) * k_n(i-2) * state((i-2)*2);
        jacobian(2*i+1, 2*i - 2) +=  (beta-1) * k_n(i-2) * state((i-2)*2);
        jacobian(2*i+1, 2*i+1 - 2) += (1-beta) * k_n(i-2) * state((i-2)*2 + 1);
    }
    
    // N
    for (int i = 0; i < dim/2; ++i) {
        jacobian(2*i, 2*i) = -nu*k_n(i)*k_n(i);
        jacobian(2*i+1, 2*i+1) = -nu*k_n(i)*k_n(i);
    }
    // std::cout << jacobian << std::endl;
    Eigen::MatrixXd Jacobian(dim, dim);
    //列の並べ替え
    for (int i = 0; i < dim/2; ++i) {
        Jacobian.col(i) = jacobian.col(2*i);
        Jacobian.col(dim/2+i) = jacobian.col(2*i+1);
    }
    Eigen::MatrixXd Jaco(dim, dim);
    //行の並べ替え
    for (int i = 0; i < dim/2; ++i) {
        Jaco.row(i) = Jacobian.row(2*i);
        Jaco.row(dim/2+i) = Jacobian.row(2*i+1);
    }
    std::cout << Jaco << std::endl;
    // Eigen::MatrixXd jacobi = npy2EigenMatR("../../jacobi.npy");
    // std::cout << jacobi - Jacobian << std::endl;
    return jacobian;
}

// Eigen::MatrixXd npy2EigenMatR(const char* fname){
//     std::string fname_str(fname);
//     cnpy::NpyArray arr = cnpy::npy_load(fname_str);
//     if (arr.word_size != sizeof(double)){
//         throw std::runtime_error("Unsupported data type in the npy file.");
//     }
//     Eigen::Map<const Eigen::MatrixXd> MatT(arr.data<double>(), arr.shape[1], arr.shape[0]);
//     return MatT.transpose();
// }