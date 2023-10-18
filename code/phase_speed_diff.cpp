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
#include "Runge_Kutta.hpp"
#include <chrono>
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname);
Eigen::VectorXcd npy2EigenVec(const char* fname);
Eigen::MatrixXcd npy2EigenMat(const char* fname);
int shift(double pre_theta, double theta, int rotation_number);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double nu = 0.00018;
    double beta = 0.4165;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 1e+4;

    double latter = 1;
    int threads = omp_get_max_threads();

    //make pairs of shells to observe phase difference(num begins from 1)
    std::vector<std::pair<int, int>> sync_pairs;

    sync_pairs.push_back(std::make_pair(4, 7));
    sync_pairs.push_back(std::make_pair(4, 10));
    sync_pairs.push_back(std::make_pair(4, 13));
    sync_pairs.push_back(std::make_pair(7, 10));
    sync_pairs.push_back(std::make_pair(7, 13));
    sync_pairs.push_back(std::make_pair(10, 13));

    sync_pairs.push_back(std::make_pair(5, 8));
    sync_pairs.push_back(std::make_pair(5, 11));
    sync_pairs.push_back(std::make_pair(5, 14));
    sync_pairs.push_back(std::make_pair(8, 11));
    sync_pairs.push_back(std::make_pair(8, 14));
    sync_pairs.push_back(std::make_pair(11, 14));

    sync_pairs.push_back(std::make_pair(6, 9));
    sync_pairs.push_back(std::make_pair(6, 12));
    sync_pairs.push_back(std::make_pair(9, 12));

    
    
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.4163_nu0.00018_10000period_dt0.01.npy");
    ShellModel solver(nu, beta, f, ddt, t_0, t, latter, x_0);
    std::cout << "calculating trajectory" << std::endl;
    Eigen::MatrixXcd trajectory = solver.get_trajectory_(); //wide matrix
    // Eigen::MatrixXcd trajectory = npy2EigenMat("../../generated_lam/generated_laminar_beta_0.417nu_0.00018_50000period1300check200progresseps0.05.npy"); //wide matrix
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
    // rotation speed
    Eigen::MatrixXd rotation_speed = (angles.bottomRows(angles.rows()-1) - angles.topRows(angles.rows()-1)) / ddt;


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
    plt::figure_size(800, 300*sync_pairs.size());
    
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
        Eigen::VectorXd diff = (rotation_speed.col(pair.first-1) - rotation_speed.col(pair.second-1)).cwiseAbs();
        for (int i = 0; i < y.size(); i++){
            y[i] = diff(i*skip);
        }
        plt::subplot(sync_pairs.size(), 1, counter);
        plt::plot(x,y);
        plt::xlabel("Time");
        plt::ylabel("$|U_{" + std::to_string(pair.first) + "}-U_{" + std::to_string(pair.second) + "}|$");
    }

    std::ostringstream oss;
    oss << "../../phase_speed_diff/beta_" << beta << "nu_" << nu <<"_"<< t-t_0 << "period";
    for (const auto& pair : sync_pairs){
        oss << "_" << std::get<0>(pair) << "-" << std::get<1>(pair);
    }
    oss << ".png";  // 文字列を結合する
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

Eigen::MatrixXcd npy2EigenMat(const char* fname){
    std::string fname_str(fname);
    cnpy::NpyArray arr = cnpy::npy_load(fname_str);
    if (arr.word_size != sizeof(std::complex<double>)){
        throw std::runtime_error("Unsupported data type in the npy file.");
    }
    Eigen::Map<const Eigen::MatrixXcd> MatT(arr.data<std::complex<double>>(), arr.shape[1], arr.shape[0]);
    return MatT.transpose();
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