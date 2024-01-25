/**
 * @file extract_sync.cpp
 * @author Hibiki Kato
 * @brief extract synchronized part of trajectory
 * @version 0.1
 * @date 2024-01-03
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <chrono> 
#include <vector>
#include <string>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"

namespace plt = matplotlibcpp;
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXcd> calc_next(ShellModel& SM, Eigen::VectorXd pre_n, Eigen::VectorXd pre_theta, Eigen::VectorXcd previous);
bool isLaminar(Eigen::VectorXd phases, std::vector<std::tuple<int, int, double>> sync_pairs);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 0.00018;
    params.beta = 0.417;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+9;
    double dump = 1e+5;
    int beta_num = 30;
    double critical = 0.416158;
    Eigen::VectorXcd x_0 = npy2EigenVec<double>("../initials/beta0.43_nu0.00018_1133period_dt0.01eps0.05.npy", true);
    // Eigen::VectorXd betas = Eigen::VectorXd::LinSpaced(beta_num, 0.416, 0.417);
    Eigen::VectorXd betas = Eigen::VectorXd::LinSpaced(beta_num, -6.2, -5.9);
    for (int i = 0; i < beta_num; i++) betas(i) = critical + std::pow(10, betas(i));
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << betas << std::endl;
    int numThreads = omp_get_max_threads();
 
    //make pairs of shells to observe phase difference(num begins from 1)
    std::vector<std::tuple<int, int, double>> sync_pairs;
    // sync_pairs.push_back(std::make_tuple(4, 7, 2.3));
    // sync_pairs.push_back(std::make_tuple(4, 10, 2.3));
    // sync_pairs.push_back(std::make_tuple(4, 13, 2.3));
    // sync_pairs.push_back(std::make_tuple(7, 10, 2));
    // sync_pairs.push_back(std::make_tuple(7, 13, 2));
    // sync_pairs.push_back(std::make_tuple(10, 13, 1E-1));

    sync_pairs.push_back(std::make_tuple(5, 8, 2.3));
    sync_pairs.push_back(std::make_tuple(5, 11, 2.3));
    sync_pairs.push_back(std::make_tuple(5, 14, 2.3));
    sync_pairs.push_back(std::make_tuple(8, 11, 0.7));
    sync_pairs.push_back(std::make_tuple(8, 14, 0.7));
    sync_pairs.push_back(std::make_tuple(11, 14, 1E-1));

    sync_pairs.push_back(std::make_tuple(6, 9, 2.3));
    sync_pairs.push_back(std::make_tuple(6, 12, 2.3));
    sync_pairs.push_back(std::make_tuple(9, 12, 0.3));

    int window = 1000; // how long the sync part should be. (sec)
    int trim = 500; // how much to trim from both starts and ends of sync part
    int skip = 1000; // period of checking sync(step)
    ShellModel SM(params, dt, t_0, t, dump, x_0);
    std::vector<double> average_durations(beta_num);
    int progress = 0;
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic) firstprivate(SM, sync_pairs, betas, window, trim, skip) shared(progress, average_durations, beta_num)
    for (int i = 0; i < beta_num; i++) {
        SM.set_beta_(betas(i));
        Eigen::VectorXcd previous = SM.x_0;
        Eigen::VectorXd n = Eigen::VectorXd::Zero(previous.rows());
        Eigen::VectorXd theta = previous.cwiseArg();
        double duration;
        std::vector<double> durations;
        // dump
        for (int j = 0; j < SM.dump_steps; j++) {
            std::tie(n, theta, previous) = calc_next(SM, n, theta, previous);
        }
        
        for (long long j = 0; j < SM.steps; j++) {
            std::tie(n, theta, previous) = calc_next(SM, n, theta, previous);
            if(j % skip == 0){
                if (isLaminar(theta, sync_pairs)){
                    duration += dt*skip;
                }else{
                    if (duration - trim*2 > window){
                        durations.push_back(duration-trim*2);
                    }
                    duration = 0;
                }
                if (omp_get_thread_num() == 0){
                    if (j % (skip * 1000) == 0) std::cout << "\r" << j*100 / SM.steps  << "%" << std::flush;
                }
            }
        }
        if (duration-trim*2 > window){
            durations.push_back(duration-trim*2);
        }
        #pragma omp atomic
        progress++;
        #pragma omp critical
        std::cout << "\r processing " << progress  << "/" << beta_num << std::flush;
        average_durations[i] = std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
        // //durationsを保存
        // std::cout << average_durations[i] << std::endl;
        // std::ostringstream oss;
        // std::ofstream ofs("durations.txt");
        // for (int j = 0; j < durations.size(); j++){
        //     ofs << durations[j] << std::endl;
        // }
        // ofs.close();
    }
    // betasとaverage_durationsを保存
    std::ostringstream oss;
    oss << "../../average_durations/data/beta" << betas(0) << "-" << betas(beta_num-1) << "_" << beta_num << "num" << "_t" << t << "_dt" << dt << "_window" << window <<".txt";
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << fname << std::endl;
    std::ofstream ofs(fname);
    for (int i = 0; i < beta_num; i++){
        ofs << std::setprecision(std::numeric_limits<double>::max_digits10) << betas(i) << " " << average_durations[i] << std::endl;
    }
    ofs.close();
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "20";
    plt::rcparams(plotSettings);

    Eigen::VectorXd betas_diff = betas.array() - critical;
    std::vector<double> betas_vec(betas_diff.data(), betas_diff.data() + betas_diff.size());

    // plot
    plt::figure_size(1200, 800);
    plt::scatter(betas_vec, average_durations, 5);
    plt::yscale("log");
    plt::xscale("log");
    plt::xlabel("beta");
    plt::ylabel("average laminar duration");

    // save
    oss.str("");
    oss << "../../average_durations/beta" << betas(0) << "-" << betas(beta_num-1) << "_" << beta_num << "num" << "_t" << t << "_dt" << dt << "_window" << window <<".png";
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    myfunc::duration(start);
}

bool isLaminar(Eigen::VectorXd phases, std::vector<std::tuple<int, int, double>> sync_pairs){
    bool allSync = true; // flag 
    for (const auto& pair : sync_pairs){
        if(! myfunc::isSync(phases(std::get<0>(pair) - 1), phases(std::get<1>(pair) - 1), std::get<2>(pair), 0)){ // if not synchronized (flag = false
            allSync = false;
            break;
        }
    }
    return allSync;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXcd> calc_next(ShellModel& SM, Eigen::VectorXd pre_n, Eigen::VectorXd pre_theta, Eigen::VectorXcd previous){
    Eigen::VectorXcd now = SM.rk4(previous);
    Eigen::VectorXd theta = now.cwiseArg();
    Eigen::VectorXd n = pre_n;
    for(int i; i < theta.size(); i++){
        n(i) = myfunc::shift(pre_theta(i), theta(i), pre_n(i));
    }
    return std::make_tuple(n, theta, now);
}