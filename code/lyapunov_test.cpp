/**
 * @file lyapunov_test.cpp
 * @author Hibiki Kato
 * @brief lyapunov exponent test
 * @version 0.1
 * @date 2023-10-04
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
void EigenVec2npy(Eigen::VectorXd Vec, std::string fname);
Eigen::VectorXcd npy2EigenVec(const char* fname);
Eigen::VectorXcd perturbation(Eigen::VectorXcd state,  std::vector<int> dim, int s_min = -1, int s_max = -1);


int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    
    double nu = 0.00018;
    double beta = 0.417;
    std::complex<double> f = std::complex<double>(1.0,0.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 2000;
    double latter = 1;
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.417_nu0.00018_7000period_dt0.002.npy");
    ShellModel SM(nu, beta, f, ddt, t_0, t, latter, x_0);
    int threads = omp_get_max_threads();
    int repetitions = 1000;
    std::cout << threads << "threads" << std::endl;
    std::ostringstream oss;

    ShellModel SM_origin = SM;
    std::vector<int> range(x_0.size());
    std::iota(range.begin(), range.end(), 1); // iota: 連番を作成する
    SM_origin.set_x_0_(perturbation(SM_origin.get_x_0_(), range, 0, 0)); // 初期値をランダムに与える
    ShellModel SM_another = SM;
    Eigen::VectorXcd perturbed_x_0 = perturbation(SM_origin.get_x_0_(), range, -4, -4); // create perturbed init value
    SM_another.set_x_0_(perturbed_x_0); // set above
    
    Eigen::MatrixXcd origin = SM_origin.get_trajectory_();
    Eigen::MatrixXcd another = SM_another.get_trajectory_();

    // Eigen::VectorXd delta_0 = (origin.topLeftCorner(origin.rows()-1, 1) - another.topLeftCorner(another.rows()-1, 1)).cwiseAbs();
    // Eigen::VectorXd delta_t = (origin.topRightCorner(origin.rows()-1, 1) - another.topRightCorner(another.rows()-1, 1)).cwiseAbs();
    // Eigen::VectorXd lyapunov = (delta_t.array() / delta_0.array()).log() / (SM.get_t_() - SM.get_t_0_());


    double delta_0 = (origin.topLeftCorner(origin.rows()-1, 1) - another.topLeftCorner(another.rows()-1, 1)).norm();
    double delta_t = (origin.topRightCorner(origin.rows()-1, 1) - another.topRightCorner(another.rows()-1, 1)).norm();
    double lyapunov = std::log(delta_t / delta_0) / (SM.get_t_() - SM.get_t_0_());
    

    std::cout << "largest lyapunov exponent: " << lyapunov << std::endl;
    

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

void EigenMt2npy(Eigen::MatrixXd Mat, std::string fname){
    Eigen::MatrixXd transposed = Mat.transpose();
    // map to const mats in memory
    Eigen::Map<const Eigen::MatrixXd> MOut(&transposed(0,0), transposed.cols(), transposed.rows());
    // save to npy file
    cnpy::npy_save(fname, MOut.data(), {(size_t)transposed.cols(), (size_t)transposed.rows()}, "w");
}

Eigen::VectorXcd perturbation(Eigen::VectorXcd state, std::vector<int> dim, int s_min, int s_max){
    Eigen::VectorXcd perturbed = state;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> s(-1, 1);
    std::uniform_real_distribution<double> dis(s_min, s_max);

    Eigen::VectorXd unit = Eigen::VectorXd::Ones(state.rows());
    for(int shell : dim){
        perturbed(shell-1) += state(shell-1) * s(gen) * std::pow(10, dis(gen)); //元の値 * (-1, 1)の一様分布 * 10^(指定の範囲から一様分布に従い選ぶ)　を雪道として与える
    }

    return perturbed;
}

void EigenVec2npy(Eigen::VectorXd Vec, std::string fname){
    std::vector<double> x(Vec.size());
    for(int i=0;i<Vec.size();i++){
        x[i]=Vec(i);
    }
    cnpy::npy_save(fname, &x[0], {(size_t)Vec.size()}, "w");
}