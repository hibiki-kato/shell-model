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
#include "Runge_Kutta.hpp"
#include <chrono>
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname);
Eigen::VectorXcd npy2EigenVec(const char* fname);
int shift(double pre_theta, double theta, int rotation_number);
bool isSync(double a, double b, double epsilon);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double nu = 0.00018;
    double beta = 0.417;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 2e+5;
    double latter = 1;
    int numthreads = omp_get_max_threads();
    int window = 1000; // how long the sync part should be. (sec)
    window *= 100; // when dt = 0.01 

    //make pairs of shells to observe phase difference(num begins from 1)
    std::vector<std::tuple<int, int, double>> sync_pairs;
    
    sync_pairs.push_back(std::make_tuple(4, 7, 2.1));
    sync_pairs.push_back(std::make_tuple(4, 10, 2.1));
    sync_pairs.push_back(std::make_tuple(4, 13, 2.1));
    sync_pairs.push_back(std::make_tuple(7, 10, 1.1));
    sync_pairs.push_back(std::make_tuple(7, 13, 1.1));
    sync_pairs.push_back(std::make_tuple(10, 13, 3.4E-2));

    sync_pairs.push_back(std::make_tuple(5, 8, 2.2));
    sync_pairs.push_back(std::make_tuple(5, 11, 2.2));
    sync_pairs.push_back(std::make_tuple(5, 14, 2.2));
    sync_pairs.push_back(std::make_tuple(8, 11, 0.55));
    sync_pairs.push_back(std::make_tuple(8, 14, 0.55));
    sync_pairs.push_back(std::make_tuple(11, 14, 8E-3));

    sync_pairs.push_back(std::make_tuple(6, 9, 1.7));
    sync_pairs.push_back(std::make_tuple(6, 12, 1.7));
    sync_pairs.push_back(std::make_tuple(9, 12, 0.2));

    // sync_pairs.push_back(std::make_tuple(1, 2, 4)); // dummy to check unextracted trajectory

    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.41616nu0.00018_1.00923e+06period.npy");
    ShellModel solver(nu, beta, f, dt, t_0, t, latter, x_0);
    std::cout << "calculating trajectory" << std::endl;
    Eigen::MatrixXcd trajectory = solver.get_trajectory_(); //wide matrix
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
            int  n= shift(angles(j-1, i), angles(j, i), rotation_number);
            // 一個前の角度に回転数を加える
            angles(j-1, i) += rotation_number * 2 * M_PI;
            // 回転数を更新
            rotation_number = n;
        }
        // 一番最後の角度に回転数を加える
        angles(angles.rows()-1, i) += rotation_number * 2 * M_PI;
    }

    std::cout << "extracting sync" << std::endl;
    std::vector<std::vector<double>> synced;
    synced.resize(angles.cols()+1);
    int counter = 0;
    for (int i = 0; i < angles.rows(); i++){
        bool allSync = true; // flag 
        for (const auto& pair : sync_pairs){
            // if any pair is not sync, allSync is false
            if(! isSync(angles(i, std::get<0>(pair)-1), angles(i, std::get<1>(pair)-1), std::get<2>(pair))){
                allSync = false;
                break;
            }
        }

        if (allSync){
            counter++;
        }
        else{
            if (counter >= window){
                //adding synchronized part to synced
                for (int j = 0 + counter/6; j < counter - 1 - counter/10; j++){
                    for (int k = 0; k < angles.cols() + 1; k++){
                        synced[k].push_back(std::abs(trajectory(k, j + i - counter)));
                    }
                }
            }
            counter = 0;
            }
    }
    //adding last part to synced
    if (counter >= window){
        for (int j = 0 + counter/6; j < counter - 1 - counter/10; j++){
            for (int k = 0; k < angles.cols() + 1; k++){
                synced[k].push_back(std::abs(trajectory(k, j + angles.rows() - counter)));
            }
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
    int skip = 100; // plot every skip points
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(800, 300);
    
    std::map<std::string, double> keywords;
    keywords.insert(std::make_pair("hspace", 0.6)); // also right, top, bottom
    keywords.insert(std::make_pair("wspace", 0.4)); // also hspace
    plt::scatter(synced[3], synced[4]);

    std::ostringstream oss;
    oss << "../../sync/beta_" << beta << "nu_" << nu <<"_"<< t-t_0 << "period" <<  window <<"window.png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

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

void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname){
    Eigen::MatrixXcd transposed = Mat.transpose();
    // map to const mats in memory
    Eigen::Map<const Eigen::MatrixXcd> MOut(&transposed(0,0), transposed.cols(), transposed.rows());
    // save to np-arrays files
    cnpy::npy_save(fname, MOut.data(), {(size_t)transposed.cols(), (size_t)transposed.rows()}, "w");
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