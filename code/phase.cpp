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
#include "Runge_Kutta.hpp"
#include <chrono>
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname);
Eigen::VectorXcd npy2EigenVec(const char* fname);
double unwrap(double pre_angle, double angle);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double nu = 0.00018;
    double beta = 0.41623;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 1e+5;
    double latter = 1;
    int numthreads = omp_get_max_threads();

    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.41616nu0.00018_1.00923e+06period.npy");
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
    int skip = 1; // plot every skip points
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(2400, 3600);
    std::vector<double> x((trajectory.cols()-1)/skip),y((trajectory.cols()-1)/skip);
    // times for x axis
    for(int i=0;i<trajectory.cols();i++){
        x[i]=trajectory.cwiseAbs()(trajectory.rows()-1, i*skip);
    }
    // plot trajectory
    for(int i=0; i < trajectory.rows()-1; i++){
        for(int j=0; j < trajectory.cols(); j++){
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
        for (int j = 0; j < angles.rows(); j++){
            if (j == 0){
                continue;
            }
            angles(j, i) = unwrap(angles(j-1, i), angles(j, i));
        }
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
        for(int j=0; j < angles.rows(); j+=skip){
            y[j]=angles.cwiseAbs()(j, i);
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
    oss << "../../phase/beta_" << beta << "nu_" << nu <<"_"<< t-t_0 << "period.png";  // 文字列を結合する
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

double unwrap(double pre_angle, double angle){
    double diff = angle - pre_angle;
    while (diff > M_PI) {
        diff -= 2 * M_PI;
        angle -= 2 * M_PI;

    }
    while (diff < -M_PI) {
        diff += 2 * M_PI;
        angle += 2 * M_PI;
    }
        return angle;
}