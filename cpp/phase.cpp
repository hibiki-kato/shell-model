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
 * @file phase.cpp
 * @author Hibiki Kato
 * @brief Calc unwrapped phase of each wavenumber of GOY shell model
 * @version 0.1
 * @date 2023-09-1
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
#include <utility> //pair用
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include <chrono>
#include "cnpy/cnpy.h"
#include "shared/matplotlibcpp.h"
namespace plt = matplotlibcpp;
void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname);
Eigen::VectorXcd npy2EigenVec<std::complex<double>>(const char* fname);
int shift(double pre_theta, double theta, int rotation_number);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 0.00018;
    params.beta = 0.5;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt =0.01;
    double t_0 = 0;
    double t = 1E+5;
    double dump =;
    int numthreads = omp_get_max_threads();

    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../../initials/beta0.417_nu0.00018_5000period_dt0.01_4-7_4-10_4-13_7-10_7-13_10-13_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy");
    ShellModel solver(nu, beta, f, ddt, t_0, t, latter, x_0);
    Eigen::MatrixXcd trajectory = solver.get_trajectory_();
    std::cout << "calculating trajectory" << std::endl;
    /*                                                                        
                  ██                                                                                                    
        ██████    ██                                                ██                                                  
        ██   ██   ██             █             █                     █                    █                             
        ██    ██  ██             █             █                                          █                             
        ██    ██  ██    ████   █████         █████  ██ ██   █████   ██    ████     ████ █████    ████    ██ ██ ██    ██ 
        ██    ██  ██   ██  ██   ██            ██    ████    █  ██   ██   ██  ██   ██  █  ██     ██  ██   ████   █    ██ 
        ██   ██   ██  ██    ██  ██            ██    ██          ██  ██  ██   ██  ██      ██    ██    ██  ██     █    █  
        ██████    ██  ██    ██  ██            ██    ██          ██  ██  ██    █  ██      ██    ██    ██  ██     ██  ██  
        ██        ██  █     ██  ██            ██    ██      ██████  ██  ███████  █       ██    █     ██  ██      █  ██  
        ██        ██  ██    ██  ██            ██    ██     ██   ██  ██  █        ██      ██    ██    ██  ██      ██ █   
        ██        ██  ██    ██  ██            ██    ██     █    ██  ██  ██       ██      ██    ██    ██  ██      ████   
        ██        ██   ██  ██    ██            ██   ██     ██  ███  ██   ██   █   ██  █   ██    ██  ██   ██       ███   
        ██        ██    ████     ███           ███  ██      ██████  ██    █████    ████   ███    ████    ██       ██    
                                                                    ██                                            ██    
                                                                    ██                                            █     
                                                                    ██                                           ██     
                                                                  ███                                          ███      
    */
    std::cout << "plotting" << std::endl;
    // plot settings
    int skip = 10; // plot every skip points
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(2400, 3600);
    std::vector<double> x((trajectory.cols()-1)/skip),y((trajectory.cols()-1)/skip);
    // times for x axis
    for(int i=0;i<x.size();i++){
        x[i]=trajectory.cwiseAbs()(trajectory.rows()-1, i*skip);
    }
    // plot trajectory
    for(int i=0; i < trajectory.rows()-1; i++){
        for(int j=0; j < y.size(); j++){
            y[j]=trajectory.cwiseAbs()(i, j*skip);
        }
        plt::subplot(trajectory.rows()-1, 2, i*2+1);
        plt::plot(x,y);
        plt::xlabel("Time");
        plt::ylabel("$U_{" + std::to_string(i+1) + "}$");
    }

    Eigen::MatrixXd angles = trajectory.topRows(trajectory.rows()-1).cwiseArg().transpose();

    std::cout << "unwrapping angles" << std::endl;

    //unwrap
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
    /*
              ██                                                       ██                 
    ██████    ██                                                       ██                 
    ██   ██   ██             █                                         ██                 
    ██    ██  ██             █                                         ██                 
    ██    ██  ██    ████   █████           █████   ██ ███      ███ ██  ██    ████    ████ 
    ██    ██  ██   ██  ██   ██             █  ██   ███  ██    ██  ███  ██   ██  ██  ██  █ 
    ██   ██   ██  ██    ██  ██                 ██  ██    ██  ██    ██  ██  ██   ██  █     
    ██████    ██  ██    ██  ██                 ██  ██    ██  ██    ██  ██  ██    █  ███   
    ██        ██  █     ██  ██             ██████  ██    ██  █     ██  ██  ███████   ████ 
    ██        ██  ██    ██  ██            ██   ██  ██    ██  ██    ██  ██  █            ██
    ██        ██  ██    ██  ██            █    ██  ██    ██  ██    ██  ██  ██           ██
    ██        ██   ██  ██    ██           ██  ███  ██    ██   ██  ███  ██   ██   █  █   ██
    ██        ██    ████     ███           ██████  ██    ██    ███ ██  ██    █████  █████ 
                                                                   █                      
                                                                   █                      
                                                              █   ██                      
                                                              █████                       
    */
    std::cout << "plotting angles" << std::endl;
    for(int i=0; i < angles.cols(); i++){
        for(int j=0; j < y.size(); j++){
            y[j]=angles(j*skip, i);
        }
        plt::subplot(angles.cols(), 2, 2*i+2);
        plt::plot(x,y);
        plt::xlabel("Time");
        plt::ylabel("$U_{" + std::to_string(i+1) + "}$");
        
    }
    
    std::map<std::string, double> keywords;
    keywords.insert(std::make_pair("hspace", 0.6)); // also right, top, bottom
    keywords.insert(std::make_pair("wspace", 0.4)); // also hspace
    plt::subplots_adjust(keywords);

    std::ostringstream oss;
    oss << "../../phase/beta" << params.beta << "nu" << params.nu <<"_"<< t-t_0 << "period.png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

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

void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname){
    Eigen::MatrixXcd transposed = Mat.transpose();
    // map to const mats in memory
    Eigen::Map<const Eigen::MatrixXcd> MOut(&transposed(0,0), transposed.cols(), transposed.rows());
    // save to np-arrays files
    cnpy::npy_save(fname, MOut.data(), {(size_t)transposed.cols(), (size_t)transposed.rows()}, "w");
}
/**
 * @brief given previous theta and rotation_number and current theta,  return rotation number(unwrapped)
 * 
 * @param pre_theta : previous theta
 * @param theta : current theta
 * @param rotation_number : previous rotation number (n in Z, unwrapped angle is theta + 2 * n * pi)
 * @return int 
 */
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