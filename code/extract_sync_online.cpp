/**
 * @file extract_sync_online.cpp
 * @author Hibiki Kato
 * @brief extract synchronized part of trajectory each time step
 * @version 0.1
 * @date 2023-09-30
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
#include <utility> // std::pair用
#include <tuple> // std::tuple用
#include <omp.h>
#include <chrono>
#include "Runge_Kutta.hpp"
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
#include "Eigen_numpy_converter.hpp"

namespace plt = matplotlibcpp;
double shift(double pre_theta, double theta, double rotation_number);
bool isLaminar(Eigen::VectorXd phases, std::vector<std::tuple<int, int, double>> sync_pairs);
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXcd> calc_next(ShellModel& SM, Eigen::VectorXd pre_n, Eigen::VectorXd pre_theta, Eigen::VectorXcd previous);
bool isSync(double a, double b, double epsilon);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
        double dt = 0.01;
        double t_0 = 0;
        double t = 1e+5;
        int numThreads = omp_get_max_threads();
        int window = 1000; // how long the sync part should be. (sec)
        window *= 100; // when dt = 0.01
        int trim = 500; 
        trim *= 100; // when dt = 0.01
        int plotDim[] = {4, 5};
        int param_num = 8;
        Eigen::VectorXd nus = Eigen::VectorXd::LinSpaced(param_num, 0.00014, 0.00017);
        Eigen::VectorXd betas = Eigen::VectorXd::LinSpaced(param_num, 0.417, 0.417);
        Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../../initials/beta0.423_nu0.00018_1229period_dt0.01eps0.003.npy");
        int skip = 100; // plot every skip points
        std::vector<std::tuple<int, int, double>> sync_pairs;

        sync_pairs.push_back(std::make_tuple(4, 7, 2.3));
        sync_pairs.push_back(std::make_tuple(4, 10, 2.3));
        sync_pairs.push_back(std::make_tuple(4, 13, 2.3));
        sync_pairs.push_back(std::make_tuple(7, 10, 2));
        sync_pairs.push_back(std::make_tuple(7, 13, 2));
        sync_pairs.push_back(std::make_tuple(10, 13, 1E-1));

        sync_pairs.push_back(std::make_tuple(5, 8, 2.3));
        sync_pairs.push_back(std::make_tuple(5, 11, 2.3));
        sync_pairs.push_back(std::make_tuple(5, 14, 2.3));
        sync_pairs.push_back(std::make_tuple(8, 11, 0.7));
        sync_pairs.push_back(std::make_tuple(8, 14, 0.7));
        sync_pairs.push_back(std::make_tuple(11, 14, 1E-1));

        sync_pairs.push_back(std::make_tuple(6, 9, 2.3));
        sync_pairs.push_back(std::make_tuple(6, 12, 2.3));
        sync_pairs.push_back(std::make_tuple(9, 12, 0.3));

        ShellModel SM = ShellModel(1e-5, 0.5, f, dt, t_0, t, 1.0, x_0);
        std::map<std::string, std::string> plotSettings;
        plotSettings["font.family"] = "Times New Roman";
        plotSettings["font.size"] = "10";
        plt::rcparams(plotSettings);

        int steps = static_cast<int>((t - t_0) / dt + 0.5);
        #pragma omp parallel for num_threads(numThreads) ordered schedule(dynamic) shared(steps, x_0, betas, nus, sync_pairs, plotDim, window, trim) firstprivate(SM)
        for (int i = 0; i < param_num; i++){
            if (omp_get_thread_num() == 0){
                std::cout << "processing " << i*numThreads << "/" << param_num << std::endl;
            }
            SM.set_beta_(betas(i));
            SM.set_nu_(nus(i));

            Eigen::VectorXd n = Eigen::VectorXd::Zero(x_0.rows());
            Eigen::VectorXd theta = SM.get_x_0_().cwiseArg();
            Eigen::VectorXcd previous = x_0;
            std::vector<double> x;
            std::vector<double> y;

            std::vector<double> synced_x;
            std::vector<double> synced_y;
            
            for (int j = 0; j < steps; j++) {
                std::tie(n, theta, previous) = calc_next(SM, n, theta, previous);
                if (isLaminar(theta+2*n*M_PI, sync_pairs)){
                    x.push_back(std::abs(previous(plotDim[0]-1)));
                    y.push_back(std::abs(previous(plotDim[1]-1)));
                }
                else{
                    if (x.size() > window){
                        synced_x.insert(synced_x.end(), x.begin(), x.end());
                        synced_y.insert(synced_y.end(), y.begin(), y.end());
                    }
                    x.clear();
                    y.clear();
                }
            }
            if (x.size() > window){
                synced_x.insert(synced_x.end(), x.begin(), x.end());
                synced_y.insert(synced_y.end(), y.begin(), y.end());
            }

            #pragma omp critical
            {
                // plot
                plt::figure_size(1200, 1200);
                std::map<std::string, std::string> plotSettings;
                plt::scatter(synced_x, synced_y);
                plt::xlim(0.0, 0.5);
                plt::ylim(0.0, 0.5);
                plt::xlabel("$|u_" + std::to_string(plotDim[0]) + "|$");
                plt::ylabel("$|u_" + std::to_string(plotDim[1]) + "|$");

                // save
                std::ostringstream oss;
                oss << "../../sync/beta_" << SM.get_beta_() << "nu_" << SM.get_nu_() <<"_"<< t-t_0 << "period" <<  static_cast<int>(window/100) <<"window" << static_cast<int>(synced_x.size()) << "sync.png";  // 文字列を結合する
                std::string plotfname = oss.str(); // 文字列を取得する
                if (synced_x.size() > 0){
                    std::cout << "Saving result to " << plotfname << std::endl;
                    plt::save(plotfname);
                }
                synced_x.clear();
                synced_y.clear();
            }
        }

        auto end = std::chrono::system_clock::now();  // 計測終了時間
        int hours = std::chrono::duration_cast<std::chrono::hours>(end-start).count(); //処理に要した時間を変換
        int minutes = std::chrono::duration_cast<std::chrono::minutes>(end-start).count(); //処理に要した時間を変換
        int seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count(); //処理に要した時間を変換
        int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間を変換
        std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;
    }

double shift(double pre_theta, double theta, double rotation_number){
    //forward
    if ((theta - pre_theta) < -M_PI){
        rotation_number += 1;
    }
    //backward
    else if ((theta - pre_theta) > M_PI){
        rotation_number -= 1;
    }

    return rotation_number;
}

bool isLaminar(Eigen::VectorXd phases, std::vector<std::tuple<int, int, double>> sync_pairs){
    bool allSync = true; // flag 
    for (const auto& pair : sync_pairs){
        if(!isSync(phases(std::get<0>(pair) - 1), phases(std::get<1>(pair) - 1), std::get<2>(pair))){
            allSync = false;
            break;
        }
    }
    return allSync;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXcd> calc_next(ShellModel& SM, Eigen::VectorXd pre_n, Eigen::VectorXd pre_theta, Eigen::VectorXcd previous){
    Eigen::VectorXcd now = SM.rk4_(previous);
    Eigen::VectorXd theta = now.cwiseArg();
    Eigen::VectorXd n = pre_n;
    for(int i; i < theta.size(); i++){
        n(i) = shift(pre_theta(i), theta(i), pre_n(i));
    }
    return std::make_tuple(n, theta, now);
}

/**
 * @brief given 2 angles, check if they are in sync
 * 
 * @param a : angle 1
 * @param b  : angle 2
 * @param epsilon : tolerance
 * @return true : sync
 * @return false : not sync
 */
bool isSync(double a, double b, double epsilon) {
    int n = 0;
    double lowerBound = 2 * n * M_PI - epsilon;
    double upperBound = 2 * n * M_PI + epsilon;
    
    while (lowerBound <= std::abs(a - b)) {
        if (lowerBound <= std::abs(a - b) && std::abs(a - b) <= upperBound) {
            // std::cout << std::abs(a-b) << std::endl;
            return true;
        }
        n++;
        lowerBound = 2 * n * M_PI - epsilon;
        upperBound = 2 * n * M_PI + epsilon;
    }
    
    return false;
}