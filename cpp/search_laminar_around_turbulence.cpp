                                                      █                                     █
█████                            █                    █     ██                              █         ██
█    █                                                █     ██                              █         ██
█    █   ███   █████ ███   ███   █  █ ███   ███    ████    ████  ███      █   █  █████   ████   ███  ████  ███
█    █  ██  █  ██  ██  █  █  ██  █  ██  █  ██  █  ██  █     ██  ██  █     █   █  ██  █  ██  █  █  ██  ██  ██  █
█████   █   █  █   █   ██     █  █  █   █  █   █  █   █     ██  █   ██    █   █  █   ██ █   █      █  ██  █   █
█   █   █████  █   █   ██  ████  █  █   █  █████  █   █     ██  █    █    █   █  █   ██ █   █   ████  ██  █████
█   ██  █      █   █   ██ █   █  █  █   █  █      █   █     ██  █   ██    █   █  █   ██ █   █  █   █  ██  █
█    █  ██  █  █   █   ██ █  ██  █  █   █  ██  █  ██  █     ██  ██  █     █   █  ██  █  ██  █  █  ██  ██  ██  █
█    ██  ████  █   █   ██ █████  █  █   █   ████   ████      ██  ███       ████  █████   ████  █████   ██  ████
                                                                                 █
                                                                                 █
                                                                                 █
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
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/matplotlibcpp.h"
#include "cnpy/cnpy.h"
namespace plt = matplotlibcpp;
Eigen::VectorXcd npy2EigenVec<std::complex<double>>(const char* fname);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    
    // generating laminar sample
    SMparams params;
    params.nu = 0.00001;
    params.beta = 0.5;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt =001;
    double t_0 = 0;
    double t = 20000;
    double dump =;
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../../initials/beta0.415_nu0.00018_100000period_dt0.01.npy");
    ShellModel SM(params, dt, t_0, t, dump, dummy);

    // set up for search
    int threads = omp_get_max_threads();
    
    int param_steps = 100;
    double beta_begin = 0.5;
    double beta_end = 0.5;
    double nu_begin = 0.0018;
    double nu_end = 0.0027;
    auto betas = Eigen::VectorXd::LinSpaced(param_steps, beta_begin, beta_end);
    Eigen::MatrixXd nus = Eigen::VectorXd::LinSpaced(param_steps, nu_begin, nu_end);
    // nus = nus.unaryExpr([](double x){return std::pow(10, x);});
    std::cout << threads << "threads" << std::endl;
    bool line = true;
    if (line != true){
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
                    int k;
                    for (k = 0; k < numRows; i++){
                        int colIdx = 10 * k;
                        traj.col(k) = trajectory.col(colIdx);
                    }
                    Eigen::VectorXd shell4 = traj.cwiseAbs().row(3);
                    Eigen::VectorXd shell5 = traj.cwiseAbs().row(4);

                    std::vector<double> Shell4(shell4.data(), shell4.data() + shell4.size());
                    std::vector<double> Shell5(shell5.data(), shell5.data() + shell5.size());
                    std::ostringstream oss;
                    oss << "../../turbulent_laminar_search/beta_" << local_SM.get_beta_() << "nu_" << local_SM.get_nu_()  << ".png";  // 文字列を結合する
                    std::string plotfname = oss.str(); // 文字列を取得する
                    #pragma omp critical
                    {
                        plt::figure_size(1000, 1000);
                        plt::xlabel("U4");
                        plt::ylabel("U5");
                        plt::plot(Shell4, Shell5);
                        plt::save(plotfname);
                        plt::close();
                    }
                }
            }
        }
    }

    else{
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
                local_SM.set_nu_(nus(i));
                auto trajectory = local_SM.get_trajectory_();
                int numRows = trajectory.cols() / 100;
                Eigen::MatrixXcd traj(trajectory.rows(), numRows);
                int j;
                for (j = 0; j < numRows; j++){
                    int colIdx = 10 * j;
                    traj.col(j) = trajectory.col(colIdx);
                }
                Eigen::VectorXd shell4 = traj.cwiseAbs().row(3);
                Eigen::VectorXd shell5 = traj.cwiseAbs().row(4);

                std::vector<double> Shell4(shell4.data(), shell4.data() + shell4.size());
                std::vector<double> Shell5(shell5.data(), shell5.data() + shell5.size());
                std::ostringstream oss;
                oss << "../../turbulent_laminar_search/beta_" << local_SM.get_beta_() << "nu_" << local_SM.get_nu_()  << ".png";  // 文字列を結合する
                std::string plotfname = oss.str(); // 文字列を取得する
                #pragma omp critical
                {
                    plt::figure_size(1000, 1000);
                    plt::xlabel("U4");
                    plt::ylabel("U5");
                    plt::xlim(0.0,0.35);
                    plt::ylim(0.0,0.5);
                    plt::plot(Shell4, Shell5);
                    plt::save(plotfname);
                    plt::close();
                }
            }
        }
    }
    myfunc::duration(start);
}

Eigen::VectorXcd npy2EigenVec<std::complex<double>>(const char* fname){
    std::string fname_str(fname);
    cnpy::NpyArray arr = cnpy::npy_load(fname_str);
    if (arr.word_size != sizeof(std::complex<double>)) {
        throw std::runtime_error("Unsupported data type in the npy file.");
    }
    std::complex<double>* data = arr.data<std::complex<double>>();
    Eigen::Map<Eigen::VectorXcd> vec(data, arr.shape[0]);
    return vec;
}