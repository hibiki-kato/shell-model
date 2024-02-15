/**
 * @file extract_sync.cpp
 * @author Hibiki Kato
 * @brief extract synchronized part of trajectory
 * @version 0.1
 * @date 2023-09-19
 *
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <utility> // std::pair用
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"
namespace plt = matplotlibcpp;

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 1.8e-4;
    params.beta = 0.417;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+6;
    double dump = 1e+4;
    int numthreads = omp_get_max_threads();
    int window = 1000; // how long the sync part should be. (sec)
    window *= 100; // 100 when dt = 0.01 
    int trim = 500; // how much to trim from both starts and ends of sync part
    trim *= 100;

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

    // sync_pairs.push_back(std::make_tuple(1, 2, 4)); // dummy to check unextracted trajectory

    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../initials/beta0.417_nu0.00018_3000period_dt0.01_4-7_4-10_4-13_7-10_7-13_10-13_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy", true);
    ShellModel SM(params, dt, t_0, t, dump, x_0);
    std::cout << "calculating trajectory" << std::endl;
    Eigen::MatrixXcd trajectory = SM.get_trajectory(); //wide matrix
    Eigen::MatrixXd angles = trajectory.topRows(trajectory.rows()-1).cwiseArg().transpose(); //tall matrix

    std::cout << "unwrapping angles" << std::endl;
    #pragma omp parallel for num_threads(numthreads)
    for (int i = 0; i < angles.cols(); i++){
        int rotation_number = 0;
        for (int j = 0; j < angles.rows(); j++){
            if (j == 0){
                continue;
            }
            //　unwrapされた角度と回転数を返す
            int  n= myfunc::shift(angles(j-1, i), angles(j, i), rotation_number);
            // 一個前の角度に回転数を加える
            angles(j-1, i) += rotation_number * 2 * M_PI;
            // 回転数を更新
            rotation_number = n;
        }
        // 一番最後の角度に回転数を加える
        angles(angles.rows()-1, i) += rotation_number * 2 * M_PI;
    }

    std::cout << "extracting sync" << std::endl;
    std::vector<std::vector<std::complex<double>>> synced;
    synced.resize(angles.cols()+1);
    int counter = 0;
    for (int i = 0; i < angles.rows(); i++){
        bool allSync = true; // flag 
        for (const auto& pair : sync_pairs){
            // if any pair is not sync, allSync is false
            if(! myfunc::isSync(angles(i, std::get<0>(pair)-1), angles(i, std::get<1>(pair)-1), std::get<2>(pair))){
                allSync = false;
                break;
            }
        }

        if (allSync){
            counter++;
        }else{
            if (counter >= window){
                //adding synchronized part to synced
                for (int j = 0 + trim; j < counter - 1 - trim; j++){
                    for (int k = 0; k < angles.cols(); k++){
                        // synced[k].push_back(trajectory(k, j + i - counter));
                        synced[k].push_back(angles(j + i - counter, k));
                    }
                    synced[14].push_back(trajectory(angles.cols(), j + i - counter));
                }
            }
            counter = 0;
            }
    }
    //adding last part to synced
    if (counter >= window){
        for (int j = 0 + counter/6; j < counter - 1 - counter/10; j++){
            for (int k = 0; k < angles.cols(); k++){
                // synced[k].push_back(trajectory(k, j + angles.rows() - counter));
                synced[k].push_back(angles(j + angles.rows() - counter, k));
            }
            synced[14].push_back(trajectory(angles.cols(), j + angles.rows() - counter));
        }
    }
    /*
            █             
    █████   █          █  
    ██  ██  █          █  
    ██   █  █   ████  ████
    ██  ██  █  ██  ██  █  
    █████   █  █    █  █  
    ██      █  █    █  █  
    ██      █  █    █  █  
    ██      █  ██  ██  ██ 
    ██      █   ████    ██
    */
    std::cout << synced[0].size() << "/" << angles.rows() << " is synchronized" <<std::endl;
    std::cout << "plotting" << std::endl;
    // plot settings
    int skip = 1; // plot every skip points
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "25";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1500, 700);
    plt::xlim(0.0, t);
    plt::xlabel("$t [sec]$");
    // plt::ylabel("$|U_{1}|$");
    plt::ylabel("$|\\phi_{6} - \\phi_{9}|$");
    std::vector<double> x_(trajectory.cols()),y_(trajectory.cols());
    for (int i = 0; i < trajectory.cols(); i++){
        x_[i] = std::abs(trajectory(14, i));
        y_[i] = std::abs(angles(i, 5) - angles(i, 8));
    }
    std::map<std::string, std::string> keywords;
    // keywords["alpha"] = "0.5";
    keywords["color"] = "lightgray";
    plt::scatter(x_, y_, 1.0, keywords);

    std::vector<double> x(synced[0].size()/skip),y(synced[0].size()/skip);
    for (int i = 0; i < y.size(); i++){
        x[i] = std::abs(synced[14][i*skip]);
        y[i] = std::abs(synced[5][i*skip] - synced[8][i*skip]);
    }
    plt::scatter(x, y, 5);

    std::ostringstream oss;
    oss << "../../sync/sync_beta" << params.beta << "nu" << params.nu <<"_"<< t-t_0 << "period" <<  window <<"window";
    for (const auto& pair : sync_pairs){
        oss << "_" << std::get<0>(pair) << "-" << std::get<1>(pair);
    }
    oss << ".png";

    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    /* 
     ████                          █
    █                              █
    █       ███  █    █   ███     ████   ████         █ ███   █ ███  █    █
    ██         █  █   █  ██  █     █    ██  ██        ██  ██  ██  ██  █   █
     ███       █  █  ██  █   ██    █    █    █        █    █  █    █  █  ██
        █   ████  ██ █   ██████    █    █    █        █    █  █    █  ██ █
        ██ █   █   █ █   █         █    █    █        █    █  █    █   █ █
        █  █   █   ███   ██        ██   ██  ██     █  █    █  ██  ██   ███
    ████   █████   ██     ████      ██   ████      ██ █    █  █████     █
                                                              █         █
                                                              █        █
                                                              █      ███
    */
    //reset oss
    oss.str("");
    oss << "../../sync/npy/sync_beta" << params.beta << "nu" << params.nu <<"_"<< t-t_0 << "period" <<  window <<"window";
    for (const auto& pair : sync_pairs){
        oss << "_" << std::get<0>(pair) << "-" << std::get<1>(pair);
    }
    oss << ".npy";

    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << fname << std::endl;
    Eigen::MatrixXcd matrix(synced.size(), synced[0].size());
    for (int i = 0; i < synced[0].size(); i++) {
        for (int j = 0; j < synced.size(); j++) {
            matrix(j, i) = synced[j][i];
        }
    }
    EigenMat2npy(matrix, fname);

    myfunc::duration(start);
}

