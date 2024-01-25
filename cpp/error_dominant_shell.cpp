/**
 * @file error_dominant_shell.cpp
 * @author Hibiki Kato
 * @brief add perturbation to 13th shell initially and calculate error ratio of each shell
 * @version 0.1
 * @date 2023-09-1
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <chrono>
#include <random>
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    
    SMparams params;
    params.nu = 1e-5;
    params.beta = 0.5;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 400;
    double dump = 0;
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("");
    ShellModel SM(params, dt, t_0, t, dump, x_0);
    int perturbed_dim = 13;
    int repetitions = 1000;
    double epsilon = 1e-5;
    int numThreads = omp_get_max_threads();
    std::cout << numThreads << "threads" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> s(-1, 1);
    Eigen::VectorXd time(SM.steps + 1); //　時間を格納するベクトル
    Eigen::MatrixXd errors(x_0.size(), SM.steps + 1);// 各試行の誤差を格納する行列
    int counter = 0; // just for progress bar
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic) firstprivate(SM, perturbed_dim, repetitions) shared(errors, time, counter)
    for(int i = 0; i < repetitions; i++){
        SM.x_0 = myfunc::multi_scale_perturbation(SM.x_0, -1, 0); // 初期値をランダムに与える
        // ある程度まともな値になるように初期値を更新
        for (int j = 0; j < 1e+5; j++) {
            SM.x_0 = SM.rk4(SM.x_0);
        }

        // ここから本番
        //まずは元の軌道を計算
        Eigen::MatrixXcd origin = SM.get_trajectory();

        // 初期値の指定した変数にだけ摂動を与える
        SM.x_0(perturbed_dim - 1) += epsilon * std::complex<double>(s(gen), s(gen));
        Eigen::MatrixXcd another = SM.get_trajectory();

        #pragma atomic
        counter++; // just for progress bar
        #pragma omp critical
        {
            errors += (origin.topRows(origin.rows() - 1) - another.topRows(another.rows() - 1)).cwiseAbs2() / repetitions;
            std::cout << "\r processing..." << counter << "/" << repetitions << std::flush;
        }
        if (i == 0) {
            time = origin.bottomRows(1).cwiseAbs().row(0);
        }
    }

    // calculate error ratio of each shell
    Eigen::VectorXd total = errors.colwise().sum();
    for (int i = 0; i < errors.cols(); i++) {
        errors.col(i) /= total(i);
    }
    // // else
    // errors = errors.cwiseSqrt();
    /*
                █              
        █████   █          █   
        █    █  █          █   
        █    █  █   ████  ████ 
        █   ██  █  ██  ██  █   
        █████   █  █    █  █   
        █       █  █    █  █   
        █       █  █    █  █   
        █       █  ██  ██  █   
        █       █   ████    ██ 
    */
    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    plt::figure_size(800, 3000);
    std::vector<double> time_vec(time.data(), time.data() + time.size());
    for(int i=0; i<errors.rows(); i+=1){
        Eigen::VectorXd ith_shell = errors.row(i);
        std::vector<double> error_vec(ith_shell.data(), ith_shell.data() + ith_shell.size());
        plt::subplot(15, 1, i+1);
        // plt::ylim(0, 1);
        plt::plot(time_vec, error_vec);
        // plt::yscale("log");
        plt::xlabel("time");
        plt::ylabel(myfunc::ordinal_suffix(i+1) + " Shell");
        error_vec.clear();
    }

    std::ostringstream oss;
    oss << "../../error_dominant_shell/beta_" << params.beta << "nu_" << params.nu << "error"<< t -t_0 <<"period" << repetitions << "repeat.png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);
    myfunc::duration(start);
}