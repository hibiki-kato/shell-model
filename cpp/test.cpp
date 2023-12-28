#include "shared/Eigen_numpy_converter.hpp"
#include <iostream>

int main(){
    Eigen::MatrixXcd mat = npy2EigenMat<std::complex<double>>("../../generated_lam/sync_gen_laminar_beta_0.418nu_0.00018_dt0.01_50000period1000check100progress10^-5-10^-3perturb_4-7_4-10_4-13_7-10_7-13_10-13_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy");
    std::cout << mat.rows() << mat.cols() << std::endl;
    
}