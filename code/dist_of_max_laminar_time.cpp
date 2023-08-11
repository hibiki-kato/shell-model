#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <algorithm>
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
    
    // generating laminar sample
    double nu = 0.0021392;
    double beta = 0.5;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.001;
    double t_0 = 0;
    double t = 10000;
    double latter = 20;
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.511579_nu0.00233572_14dim_period.npy");
    ShellModel SM(nu, beta, f, ddt, t_0, t, latter, x_0);
    Eigen::MatrixXcd laminar = SM.get_trajectory_();
    int numRows = laminar.cols() / 1000;
    Eigen::MatrixXcd laminar_sample(laminar.rows(), numRows);
    for (int i = 0; i < numRows; i++){
        int colIdx = 1000 * i;
        laminar_sample.col(i) = laminar.col(colIdx);
    }
    double beta_of_laminar = beta;

    // set up for search
    double dump = 0;
    t=1e+5;
    latter = 1;
    int skip = 100000;
    double epsilon = 1E-1;
    int threads = omp_get_max_threads();
    std::cout << threads << "threads" << std::endl;

    LongLaminar LL(nu, beta, f, ddt, t_0, t, latter, x_0, laminar_sample, epsilon, skip, 100, 10, threads);
    
    int repetitions = 1;
    int param_steps = 96;
    double beta_begin = 0.5;
    double beta_end = 0.5;
    double nu_begin = 0.0019;
    double nu_end = 0.0026;
    auto betas = Eigen::VectorXd::LinSpaced(param_steps, beta_begin, beta_end);
    auto nus = Eigen::VectorXd::LinSpaced(param_steps, nu_begin, nu_end);
    bool line = true;
    std::ostringstream oss;
    Eigen::MatrixXd result;

    if (!line){
        // 交差的にO(n^2)でやる場合。
        result.resize(param_steps * param_steps, 3);
        #pragma omp parallel for num_threads(threads)
        for(int i = 0; i < param_steps; i++){
            if (omp_get_thread_num() ==0){
                std::cout << "\r 現在" << i * threads << "/" << param_steps << std::flush;
            }
            LongLaminar local_LL = LL;
            LongLaminar LL_for_dump = LL;
            local_LL.set_beta_(betas(i));
            int j;
            for(j = 0; j < param_steps; j++){
                local_LL.set_nu_(nus(j));
                LL_for_dump = local_LL;
                LL_for_dump.set_t_(dump);
                double maxtime = 0;
                // 指定回数同じパラメータで計算して平均を取る
                int k;
                for(k = 0; k < repetitions; k++){
                    LL_for_dump.set_x_0_(LL_for_dump.perturbation_(LL_for_dump.get_x_0_()));
                    local_LL.set_x_0_(LL_for_dump.get_trajectory_().topRightCorner(14, 1));
                    std::vector<double> durations = local_LL.laminar_duration_();
                    maxtime += *std::max_element(durations.begin(), durations.end());
                }
                #pragma omp critical
                result.row(param_steps * i + j) << betas(i), nus(j), maxtime / repetitions;
            }
        }
        oss << std::defaultfloat << "../../max_time_para/max_laminar_time_beta" << beta_begin <<"to"<< beta_end << "_nu" << nu_begin <<"to" << nu_end <<"_"<< param_steps << "times_epsilon" << epsilon << "_" << t-t_0 << "period" << dump << "dump" << repetitions <<"repeat_laminar"<< beta_of_laminar << ".npy";  // 文字列を結合する
    }
    else{
        // beta nuを同時に動かす O(n)の場合
        result.resize(param_steps, 3);
        #pragma omp parallel for num_threads(threads)
        for(int i = 0; i < param_steps; i++){
            if (omp_get_thread_num() ==0){
                std::cout << "\r 現在" << i*threads << "/" << param_steps << std::flush;
            }
            LongLaminar local_LL = LL;
            LongLaminar LL_for_dump = LL;
            local_LL.set_beta_(betas(i));
            local_LL.set_nu_(nus(i));
            double maxtime = 0;
            // 指定回数同じパラメータで計算して平均を取る
            int k;
            for(k = 0; k < repetitions; k++){
                LL_for_dump.set_x_0_(LL_for_dump.perturbation_(LL_for_dump.get_x_0_()));
                local_LL.set_x_0_(LL_for_dump.get_trajectory_().topRightCorner(14, 1));
                std::vector<double> durations = local_LL.laminar_duration_();
                maxtime += *std::max_element(durations.begin(), durations.end());
            }
            #pragma omp critical
            result.row(i) << betas(i), nus(i), maxtime / repetitions;
            }
        oss << std::defaultfloat << "../../max_time_para/max_laminar_time(line)_beta" << beta_begin <<"to"<< beta_end << "_nu" << nu_begin <<"to" << nu_end <<"_"<< param_steps << "times_epsilon" << epsilon << "_" << t-t_0 << "period" << dump << "dump" << repetitions <<"repeat_laminar"<< beta_of_laminar << ".npy";  // 文字列を結合する
    }
    


    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "saving as . . ." << fname << std::endl;
    EigenMt2npy(result, fname);

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
    // save to np-arrays files
    cnpy::npy_save(fname, MOut.data(), {(size_t)transposed.cols(), (size_t)transposed.rows()}, "w");
}