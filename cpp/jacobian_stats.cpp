/*
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
*/
/**
 * @file average_jacobian.cpp
 * @author hibiki kato
 * @brief calculate average jacobian matrix and calc trajectory by using it. The aim is to see the mechanism of error growth by jacobian.
 * @version 0.1
 * @date 2023-08-15
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
#include <eigen3/Eigen/Dense>
#include "cnpy/cnpy.h"
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/matplotlibcpp.h"
namespace plt = matplotlibcpp;

// メイン関数
int main() {
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 0.00001;
    params.beta = 0.5;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 10000;
    double dump =
    int threads = omp_get_max_threads();
    Eigen::VectorXcd dummy = Eigen::VectorXd::Zero(15);
    const char* fname = "../../beta0.5_nu1e-05_40000period.npy";
    
    double repetitions = 1;
    double r = 1E-5;
    ShellModel SM(params, dt, t_0, t, latter, dummy);
    
    std::cout << "loading trajectory" << std::endl;
    Eigen::MatrixXd Data = npy2RealMat(fname);
    int numTimeSteps = Data.cols();
    int numVariables = Data.rows();
    int dim = numVariables / 2;
    // Eigen::VectorXd time = npy2EigenMat(fname).col(dim).real();

    // 斜めの要素9本ごとに統計を見る
    Eigen::MatrixXd upperBelt5(numTimeSteps, dim*2-5);
    Eigen::MatrixXd upperBelt4(numTimeSteps, dim*2-4);
    Eigen::MatrixXd upperBelt3(numTimeSteps, dim*2-3);
    Eigen::MatrixXd upperBelt2(numTimeSteps, dim*2-2);
    Eigen::MatrixXd upperBelt1(numTimeSteps, dim*2-1);
    Eigen::MatrixXd lowerBelt1(numTimeSteps, dim*2-1);
    Eigen::MatrixXd lowerBelt2(numTimeSteps, dim*2-2);
    Eigen::MatrixXd lowerBelt3(numTimeSteps, dim*2-3);
    Eigen::MatrixXd lowerBelt4(numTimeSteps, dim*2-4);
    Eigen::MatrixXd lowerBelt5(numTimeSteps, dim*2-5);

    std::cout << "calculating jacobian" << std::endl;
    // ヤコビ行列の計算
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < numTimeSteps; ++i) {
        VectorXd state = Data.col(i);
        // ヤコビアンの計算
        auto jacobian = computeJacobian(state, SM.get_k_n_(), SM.get_beta_(), SM.get_nu_());

        //　ヤコビアンの斜め要素を行列に格納
        for (int j = 0; j < dim*2-5; j++){
            upperBelt5(i, j) = jacobian(j, j+5);
            lowerBelt5(i, j) = jacobian(j+5, j);
        }

        for (int j = 0; j < dim*2-4; j++){
            upperBelt4(i, j) = jacobian(j, j+4);
            lowerBelt4(i, j) = jacobian(j+4, j);
        }

        for (int j = 0; j < dim*2-3; j++){
            upperBelt3(i, j) = jacobian(j, j+3);
            lowerBelt3(i, j) = jacobian(j+3, j);
        }

        for (int j = 0; j < dim*2-2; j++){
            upperBelt2(i, j) = jacobian(j, j+2);
            lowerBelt2(i, j) = jacobian(j+2, j);
        }

        for (int j = 0; j < dim*2-1; j++){
            upperBelt1(i, j) = jacobian(j, j+1);
            lowerBelt1(i, j) = jacobian(j+1, j);
        }

    }
    

    
    // 結果の表示
    std::cout << "plotting" << std::endl;
    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(3000, 3000);
    int skip = 100;
    std::vector<double> x((numTimeSteps-1)/skip),y1((numTimeSteps-1)/skip), y2((numTimeSteps-1)/skip);
    // times for x axis
    // for(int i=0;i<trajectory.cols();i++){
    //     x[i]=time(i*skip);
    // }

    // plot hist of each belt
    for(int i=0; i < dim*2-5; i++){
        for(int j=0; j < y1.size(); j++){
            y1[j]=upperBelt5(j*skip, i);
            y2[j]=lowerBelt5(j*skip, i);
        }
        plt::subplot(dim*2, dim*2, i*dim*2 + i + 5 + 1);
        plt::hist(y1);

        plt::subplot(dim*2, dim*2, (i+5)*dim*2 + i + 1);
        plt::hist(y2);
    }
    std::cout << "first done" << std::endl;
    for(int i=0; i < dim*2-4; i++){
        for(int j=0; j < y1.size(); j++){
            y1[j]=upperBelt4(j*skip, i);
            y2[j]=lowerBelt4(j*skip, i);
        }
        plt::subplot(dim*2, dim*2, i*dim*2 + i + 4 + 1);
        plt::hist(y1);

        plt::subplot(dim*2, dim*2, (i+4)*dim*2 + i + 1);
        plt::hist(y2);
    }
    std::cout << "second done" << std::endl;
    for(int i=0; i < dim*2-3; i++){
        for(int j=0; j < y1.size(); j++){
            y1[j]=upperBelt3(j*skip, i);
            y2[j]=lowerBelt3(j*skip, i);
        }
        plt::subplot(dim*2, dim*2, i*dim*2 + i + 3 + 1);
        plt::hist(y1);

        plt::subplot(dim*2, dim*2, (i+3)*dim*2 + i + 1);
        plt::hist(y2);
    }
    std::cout << "third done" << std::endl;
    for(int i=0; i < dim*2-2; i++){
        for(int j=0; j < y1.size(); j++){
            y1[j]=upperBelt2(j*skip, i);
            y2[j]=lowerBelt2(j*skip, i);
        }
        plt::subplot(dim*2, dim*2, i*dim*2 + i + 2 + 1);
        plt::hist(y1);

        plt::subplot(dim*2, dim*2, (i+2)*dim*2 + i + 1);
        plt::hist(y2);
    }

    for(int i=0; i < dim*2-1; i++){
        for(int j=0; j < y1.size(); j++){
            y1[j]=upperBelt1(j*skip, i);
            y2[j]=lowerBelt1(j*skip, i);
        }
        plt::subplot(dim*2, dim*2, i*dim*2 + i + 1 + 1);
        plt::hist(y1);

        plt::subplot(dim*2, dim*2, (i+1)*dim*2 + i + 1);
        plt::hist(y2);
    }

    std::map<std::string, double> keywords;
    keywords.insert(std::make_pair("hspace", 0.5)); // also right, top, bottom
    keywords.insert(std::make_pair("wspace", 0.5)); // also hspace
    plt::subplots_adjust(keywords);

    std::ostringstream oss;
    oss << "../../jacobian/dist_beta" << params.beta << "nu" << params.nu <<"_"<< t-t_0 << "period"<<repetitions << "repeat.png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    oss.str("");
     // 文字列を取得する
    oss << "../../beta" << beta << "_nu" << nu <<"_"<< t-t_0 << "period.npy";  // 文字列を結合する
    std::string npyfname = oss.str();
    // std::cout << "Saving result to " << npyfname << std::endl;
    // EigenMt2npy(traj, npyfname);

    

    myfunc::duration(start);
    return 0;
}