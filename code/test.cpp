#include <eigen3/Eigen/Dense>
#include <complex>
#include <iostream>
#include <vector>
#include <random>
#include <type_traits>
// #include <cmath>
#include "matplotlibcpp.h"
#include "cnpy/cnpy.h"
namespace plt = matplotlibcpp;
Eigen::MatrixXcd npy2EigenMat(const char* fname);
Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& state, Eigen::VectorXd k_n, double beta, double nu);

int main(){
    double nu = 0.00001;
    double beta = 0.5;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 10000;
    double latter = 1;
    Eigen::VectorXcd x_0(14);
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

    Eigen::VectorXd k_n(14);
    for (int i = 0; i < 14; ++i) {
        k_n(i) = pow(2, i-3);
        x_0(i) = std::complex<double>(1, 1);
    }
    Eigen::VectorXd state(28);
    for (int i = 0; i < 14; ++i) {
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
    // std::cout << jacobian << std::endl;
    jacobian = Eigen::MatrixXd::Zero(dim, dim);
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
    // std::cout << jacobian << std::endl;
    jacobian = Eigen::MatrixXd::Zero(dim, dim);

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
    
    // std::cout << jacobian << std::endl;
    jacobian = Eigen::MatrixXd::Zero(dim, dim);

    // N
    for (int i = 0; i < dim/2; ++i) {
        jacobian(2*i, 2*i) = -nu*k_n(i)*k_n(i);
        jacobian(2*i+1, 2*i+1) = -nu*k_n(i)*k_n(i);
    }

    std::cout << jacobian << std::endl;

    return jacobian;
}