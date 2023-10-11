/**
 * @file stagger_and_step_sync.cpp
 * @author Hibiki Kato
 * @brief Conduct stagger and step method using synchronization
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
#include <tuple>
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
    double check = 1e+3;
    double progress = 100;
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.41616nu0.00018_1.00923e+06period.npy");
    ShellModel SM(nu, beta, f, dt, t_0, t, latter, x_0);
    int numthreads = omp_get_max_threads();

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

    // sync_pairs.push_back(std::make_tuple(1, 2, 4)); // dummy to check all trajectory

    /*
      ██                                                                    █
    █████                                                                   █
    █      ██                                                               █            ██
    █     ████   ████   █████   █████   ████   █ ██      ████  ██████   █████     █████ ████   ████   █████
    ███    █        █  ██  ██  ██  ██  █   ██  ██           █  ██   █  ██  ██     █      █    █   ██  ██   █
      ███  █        █  █    █  █    █  █   ██  █            █  █    █  █    █     ██     █    █   ██  █    █
        ██ █    █████  █    █  █    █  ██████  █        █████  █    █  █    █      ███   █    ██████  █    █
        ██ █    █   █  █    █  █    █  █       █        █   █  █    █  █    █        ██  █    █       █    █
        █  ██   █   █  ██  ██  ██  ██  ██      █        █   █  █    █  ██  ██        ██  ██   ██      ██   █
    █████   ███ █████   █████   █████   ████   █        █████  █    █   ███ █     ████    ███  ████   █████
                            █       █                                                                █
                           ██      ██                                                                █
                       █████   █████                                                                 █
    */
    Eigen::MatrixXcd calced_laminar(x_0.rows()+1, SM.get_steps_()+1);
    int stagger_and_step_num = static_cast<int>((t-t_0) / progress + 0.5);
    int check_steps = static_cast<int>(check / dt + 0.5);
    int progress_steps = static_cast<int>(progress / dt + 0.5);
    SM.set_steps_(check_steps);

    for (int i; i < stagger_and_step_num; i++){
        std::cout << "\r 現在" << SM.get_t_0_() << "時間" << std::flush;
        bool laminar = true; // flag
        double now_time = SM.get_t_0_();
        Eigen::VectorXcd now = SM.get_x_0_();
        Eigen::MatrixXcd trajectory = Eigen::MatrixXcd::Zero(x_0.rows()+1, progress_steps+1); //wide matrix for progress
        trajectory.topLeftCorner(x_0.size(), 1) = now;
        // SM.rk4_()　のポインターを作成
        Eigen::VectorXcd (*rk4)(Eigen::VectorXcd) = SM.rk4_;

        // no perturbation at first
        Eigen::VectorXi n = Eigen::VectorXi::Zero(x_0.rows());
        Eigen::VectorXd theta = trajectory.topRows(trajectory.rows()-1).cwiseArg().transpose();
        for (int j = 0; j < check_steps; j++){
            std::tuple<Eigen::VectorXi, Eigen::VectorXd, Eigen::VectorXcd> next = SM.calc_next_(rk4, n, theta, trajectory);
            n = std::get<0>(next);
            theta = std::get<1>(next);
            now = std::get<2>(next);
            if (isLaminar(theta, sync_pairs)){
                if (j < progress_steps){
                    not_time += dt;
                    trajectory.block(0, j+1, x_0.rows()-1, j+1) = now;
                    trajectory(x_0.rows(), j+1) = ;
                }
            }
            else{
                laminar = false;
                break;
            }
        }
        // if laminar, continue to for loop
        if (laminar){
            SM.set_t_0_(now_time);
            SM.set_x_0_(now);
            calced_laminar.block(0, i*progress_steps, x_0.rows(), progress_steps) = trajectory.topLeftCorner(x_0.rows(), progress_steps);
            calced_laminar(x_0.rows(), i*progress_steps) = trajectory(x_0.rows(), progress_steps);
            continue;
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
    std::vector<double> x(calced_laminar.cols()),y(calced_laminar.cols());
    int reach = static_cast<int>(calced_laminar.bottomRightCorner(1, 1).cwiseAbs()(0, 0) + 0.5); 

    for(int i=0;i<calced_laminar.cols();i++){
        x[i]=calced_laminar.cwiseAbs()(3, i);
        y[i]=calced_laminar.cwiseAbs()(4, i);
    }

    plt::plot(x,y);
    std::ostringstream oss;
    oss << "../../generated_lam_imag/sync_gen_laminar_beta_" << beta << "nu_" << nu <<"_"<< reach << "period.png";  // 文字列を結合する
    std::string filename = oss.str(); // 文字列を取得する
    std::cout << "\n Saving result to " << filename << std::endl;
    plt::save(filename);

    oss.str("");
    oss << "../../generated_lam/sync_gen_laminar_beta_" << beta << "nu_" << nu <<"_dt"<< dt << "_" << reach << "period" << check_sec << "check" << progress_sec << "progress" << "eps" << epsilon << ".npy";
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "saving as " << fname << std::endl;
    EigenMt2npy(calced_laminar, fname);

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

bool isLaminar(Eigen::VectorXd phases, std::vector<std::tuple<int, int, double>> sync_pairs){
    bool allSync = true; // flag 
    for (const auto& pair : sync_pairs){
        // if any pair is not sync, allSync is false
        if(std::abs(phases(std::get<0>(pair)-1) - phases(std::get<1>(pair)-1)) > std::get<2>(pair)){
            allSync = false;
            break;
        }
    }
    return allSync;
}

std::tuple<Eigen::VectorXi, Eigen::VectorXd> calc_next(Eigen::VectorXcd (*rk4)(Eigen::VectorXcd), Eigen::VectorXi pre_n, Eigen::VectorXd pre_theta, Eigen::MatrixXcd previous){
    Eigen::VectorXcd now = rk4(previous);
    Eigen::VectorXd theta = now.cwiseArg();
    Eigen::VectorXi n = pre_n;
    for(int i; i < theta.size(); i++){
        n(i) = shift(pre_theta(i), theta(i), pre_n(i));
    }
    return std::make_tuple(n, theta);
}
