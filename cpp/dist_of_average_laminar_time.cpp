/*
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
*/
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
#include <omp.h>
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/matplotlibcpp.h"
#include "cnpy/cnpy.h"
namespace plt = matplotlibcpp;

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    
    // generating laminar sample
    SMparams params;
    params.nu = 0.00018;
    params.beta = 0.417;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.001;
    double t_0 = 0;
    double t = 10000;
    double dump = 20;
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>("../../initials/beta0.416_nu0.00017520319481270297_step0.01_10000.0period_laminar.npy");
    ShellModel SM(params, dt, t_0, t, dump, dummy);
    Eigen::MatrixXcd laminar = SM.get_trajectory_();
    int numRows = laminar.cols() / 100;
    Eigen::MatrixXcd laminar_sample(laminar.rows(), numRows);
    for (int i = 0; i < numRows; i++){
        int colIdx = 100 * i;
        laminar_sample.col(i) = laminar.col(colIdx);
    }
    // preserve beta of laminar
    auto beta_of_laminar = beta;

    // set up for search
    t=1E+5;
    double dump = 1e+3;
    double floor_threshold = 0;
    latter = 1;
    int skip = 10000;
    double epsilon = 1E-1;
    int threads = omp_get_max_threads();
    
    int param_steps = 96;
    int repetitions = 1;
    double beta_begin = 0.5;
    double beta_end = 0.5;
    double nu_begin = 0.0019;
    double nu_end = 0.0025;
    auto betas = Eigen::VectorXd::LinSpaced(param_steps, beta_begin, beta_end);
    auto nus = Eigen::VectorXd::LinSpaced(param_steps, nu_begin, nu_end);
    std::cout << threads << "threads" << std::endl;
    LongLaminar LL(nu, beta, f, ddt, t_0, t, latter, x_0, laminar_sample, epsilon, skip, 100, 10, threads);
    std::ostringstream oss;
    Eigen::MatrixXd result;

        // 交差的にO(n^2)でやる場合。
        result.resize(param_steps * param_steps, 3);
        #pragma omp parallel for num_threads(threads)
        for(int i = 0; i < param_steps; i++){
            if (omp_get_thread_num() ==0){
                std::cout << "\r 現在" << i * threads << "/" << param_steps << std::flush;
            }
            LongLaminar local_LL = LL;
            LongLaminar LL_for_dump = local_LL;
            local_LL.set_beta_(betas(i));
            for(int j = 0; j < param_steps; j++){
                Eigen::VectorXcd perturbed_x_0 = local_LL.perturbation_(local_LL.get_x_0_());
                local_LL.set_x_0_(perturbed_x_0);
                local_LL.set_nu_(nus(j));
                LL_for_dump = local_LL;
                LL_for_dump.set_t_(dump);
                double average_time = 0;
                // 指定回数同じパラメータで計算して平均を取る
                for(int k = 0; k < repetitions; k++){
                    LL_for_dump.set_x_0_(LL_for_dump.perturbation_(LL_for_dump.get_x_0_()));
                    local_LL.set_x_0_(LL_for_dump.get_trajectory_().topRightCorner(14, 1));
                    std::vector<double> durations = local_LL.laminar_duration_();
                    std::vector<double> filtered;
                    std::copy_if(durations.begin(), durations.end(), std::back_inserter(filtered), [](double x){ return x > 1e+4; });

                    average_time += std::accumulate(durations.begin(), filtered.end(), 0.0) / filtered.size();
                }
                #pragma omp critical
                result.row(param_steps * i + j) << betas(i), nus(j), average_time / repetitions;
            }
        }
        oss << std::defaultfloat << "../../average_time_para/average_laminar_time_beta" << beta_begin <<"to"<< beta_end << "_nu" << nu_begin <<"to" << nu_end <<"_"<< std::scientific << param_steps << "times_epsilon" << epsilon << "_" << t-t_0 << "period" << dump << "dump" << repetitions <<"repeat_floor"<< floor_threshold << "laminar"<< beta_of_laminar << ".npy";  // 文字列を結合する
    }
    
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "saving as . . ." << fname << std::endl;
    EigenMt2npy(result, fname);

    myfunc::duration(start);
}