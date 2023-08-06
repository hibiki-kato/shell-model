#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <iterator>
#include <complex>
#include <cmath>
#include <chrono>
#include <random>
#include <omp.h>
#include "Runge_Kutta.hpp"
#include "matplotlibcpp.h"
#include "cnpy/cnpy.h"
namespace plt = matplotlibcpp;
Eigen::VectorXcd npy2EigenVec(const char* fname);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    
    // generating laminar sample
    double nu = 0.00001;
    double beta = 0.5;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 10000;
    double latter = 4;
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.416_nu0.00017520319481270297_step0.01_10000.0period_laminar.npy");
    ShellModel SM(nu, beta, f, ddt, t_0, t, latter, x_0);

    // set up for search
    int threads = omp_get_max_threads();
    
    int param_steps = 100;
    int repetitions = 1;
    double beta_begin = 4.8e-01;
    double beta_end = 5.2e-01;
    double nu_begin = -3;
    double nu_end = -8;
    auto betas = Eigen::VectorXd::LinSpaced(param_steps, beta_begin, beta_end);
    Eigen::MatrixXd nus = Eigen::VectorXd::LinSpaced(param_steps, nu_begin, nu_end);
    nus = nus.unaryExpr([](double x){return std::pow(10, x);});
    std::cout << threads << "threads" << std::endl;

    #pragma omp parallel num_threads(threads)
    {
        
        int counter = 0;
        #pragma omp for
        for(int i = 0; i < param_steps; i++){
            ShellModel local_SM = SM;
            if (omp_get_thread_num() ==0){
                std::cout << "\r 現在" << counter * threads << "/" << param_steps << std::flush;
                counter++;
            }
            local_SM.set_beta_(betas(i));
            int j;
            for(j = 0; j < param_steps; j++){
                local_SM.set_nu_(nus(j));
                auto trajectory = local_SM.get_trajectory_();
                int numRows = trajectory.cols() / 100;
                Eigen::MatrixXcd traj(trajectory.rows(), numRows);
                for (int i = 0; i < numRows; i++){
                    int colIdx = 10 * i;
                    traj.col(i) = trajectory.col(colIdx);
    }
                Eigen::VectorXd shell4 = traj.cwiseAbs().row(3);
                Eigen::VectorXd shell5 = traj.cwiseAbs().row(4);

                std::vector<double> Shell4(shell4.data(), shell4.data() + shell4.size());
                std::vector<double> Shell5(shell5.data(), shell5.data() + shell5.size());
                plt::figure_size(1000, 1000);
                plt::xlabel("U4");
                plt::ylabel("U5");
                plt::plot(Shell4, Shell5);
                std::ostringstream oss;
                oss << "../../turbulent_laminar_search/beta_" << local_SM.get_beta_() << "nu_" << local_SM.get_nu_()  << ".png";  // 文字列を結合する
                std::string plotfname = oss.str(); // 文字列を取得する
                #pragma omp critical
                {
                plt::save(plotfname);
                plt::close();
                }
            }
        }
    }

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