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
/**
 * @file phase_diff.cpp
 * @author Hibiki Kato
 * @brief Observe phase difference between shells
 * @version 0.1
 * @date 2023-09-19
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
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include <chrono>
#include "cnpy/cnpy.h"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"

namespace plt = matplotlibcpp;
int shift(double pre_theta, double theta, int rotation_number);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 0.00018;
    params.beta = 0.5;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt =.01;
    double t_0 = 0;
    double t = 1e+5;

    double dump =
    int threads = omp_get_max_threads();
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../../initials/beta0.417_nu0.00018_5000period_dt0.01_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy");

    //make pairs of shells to observe phase difference(num begins from 1)
    std::vector<std::pair<int, int>> sync_pairs;

    // sync_pairs.push_back(std::make_pair(4, 7));
    // sync_pairs.push_back(std::make_pair(4, 10));
    // sync_pairs.push_back(std::make_pair(4, 13));
    // sync_pairs.push_back(std::make_pair(7, 10));
    // sync_pairs.push_back(std::make_pair(7, 13));
    // sync_pairs.push_back(std::make_pair(10, 13));

    // sync_pairs.push_back(std::make_pair(5, 8));
    // sync_pairs.push_back(std::make_pair(5, 11));
    // sync_pairs.push_back(std::make_pair(5, 14));
    // sync_pairs.push_back(std::make_pair(8, 11));
    // sync_pairs.push_back(std::make_pair(8, 14));
    // sync_pairs.push_back(std::make_pair(11, 14));

    sync_pairs.push_back(std::make_pair(6, 9));
    // sync_pairs.push_back(std::make_pair(6, 12));
    // sync_pairs.push_back(std::make_pair(9, 12));

    ShellModel solver(nu, beta, f, ddt, t_0, t, latter, x_0);
    std::cout << "calculating trajectory" << std::endl;
    Eigen::MatrixXcd trajectory = solver.get_trajectory_(); //wide matrix
    // Eigen::MatrixXcd trajectory = npy2EigenMat<std::complex<double>>("../../generated_lam/sync_gen_laminar_beta_0.43nu_0.00018_dt0.01_5000period3000check100progress10^-8-10^-5perturb_4-7_4-10_4-13_7-10_7-13_10-13_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy"); //wide matrix
    // Eigen::MatrixXcd trajectory = trajectory_.leftCols(500000);
    Eigen::MatrixXd angles = trajectory.topRows(trajectory.rows()-1).cwiseArg().transpose(); //tall matrix

    std::cout << "unwrapping angles" << std::endl;
    //unwrap
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < angles.cols(); i++){
        int rotation_number = 0;
        for (int j = 0; j < angles.rows(); j++){
            if (j == 0){
                continue;
            }
            //　unwrapされた角度と回転数を返す
            int  n= shift(angles(j-1, i), angles(j, i), rotation_number);
            // 一個前の角度に回転数を加える
            angles(j-1, i) += rotation_number * 2 * M_PI;
            // 回転数を更新
            rotation_number = n;
        }
        // 一番最後の角度に回転数を加える
        angles(angles.rows()-1, i) += rotation_number * 2 * M_PI;
    }


    /*
            █                       █  █   ███ ███
    █████   █          █            █      █   █  
    ██  ██  █          █            █     ██  ██  
    ██   █  █   ████  ████      █████  █ ████████ 
    ██  ██  █  ██  ██  █       ██  ██  █  ██  ██  
    █████   █  █    █  █       █    █  █  ██  ██  
    ██      █  █    █  █       █    █  █  ██  ██  
    ██      █  █    █  █       █    █  █  ██  ██  
    ██      █  ██  ██  ██      ██  ██  █  ██  ██  
    ██      █   ████    ██      ███ █  █  ██  ██  
    */

    std::cout << "plotting" << std::endl;
    // plot settings
    int skip = 1; // plot every skip points
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1200, 400*sync_pairs.size());
    
    std::map<std::string, double> keywords;
    keywords.insert(std::make_pair("hspace", 0.6)); // also right, top, bottom
    keywords.insert(std::make_pair("wspace", 0.4)); // also hspace
    plt::subplots_adjust(keywords);

    std::vector<double> x((trajectory.cols()-1)/skip),y((trajectory.cols()-1)/skip);

    // times for x axis
    for(int i=0;i<x.size();i++){
        x[i]=trajectory.cwiseAbs()(trajectory.rows()-1, i*skip);
    }
    int counter = 0;
    for(const auto& pair : sync_pairs){
        counter++;
        Eigen::VectorXd diff = (angles.col(pair.first-1) - angles.col(pair.second-1)).cwiseAbs();
        for (int i = 0; i < y.size(); i++){
            y[i] = diff(i*skip);
        }
        plt::subplot(sync_pairs.size(), 1, counter);
        // 2pi から 12pi　までを2piずつhlineで薄く横線をひく
        for (int i = 0; i < 7; i++){
            plt::axhline(i*2*M_PI, 0.0, 1e+5, {{"color", "gray"}, {"linestyle", ":"}});
        }
        plt::plot(x,y);
        plt::xlabel("t [sec]");
        plt::ylabel("$|\\phi_" + std::to_string(pair.first) + "-\\phi_" + std::to_string(pair.second) + "|$");
    }

    std::ostringstream oss;
    oss << "../../phase_diff/beta" << params.beta << "nu" << params.nu <<"_"<< t-t_0 << "period";
    for (const auto& pair : sync_pairs){
        oss << "_" << std::get<0>(pair) << "-" << std::get<1>(pair);
    }
    oss << ".png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    myfunc::duration(start);
}

int shift(double pre_theta, double theta, int rotation_number){
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