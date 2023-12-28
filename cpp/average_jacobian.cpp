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
#include <numeric>
#include <omp.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "cnpy/cnpy.h"
#include "shared/Flow.hpp"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/myFunc.hpp"
namespace plt = matplotlibcpp;

// メイン関数
int main() {
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 1e-5;
    params.beta = 0.5;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+3;
    double dump = 0;
    Eigen::VectorXcd dummy = Eigen::VectorXd::Zero(15);
    ShellModel SM(params, dt, t_0, t, dump, dummy);
    
    double repetitions = 1;
    int numThreads = 1;
    int error_add_dim = 13;
    double r = 1E-5; // 初期エラーの大きさ

    // データは読み込み必須
    Eigen::MatrixXcd rawData = npy2EigenMat<std::complex<double>>("../traj/beta0.5_nu1e-05_1000period.npy", true);
    Eigen::MatrixXd Time = rawData.row(rawData.rows()-1).cwiseAbs();
    // パラメータの設定（例）
    int dim = rawData.rows() - 1;
    // データの整形(実関数化)
    Eigen::MatrixXd Data(dim*2, rawData.cols());
    for (int i = 0; i < dim; ++i) {
        Data.row(2*i) = rawData.row(i).real();
        Data.row(2*i+1) = rawData.row(i).imag();
    }

    int numTimeSteps = Data.cols();
    int numVariables = Data.rows();
    Eigen::MatrixXd average_jacobian(dim*2, dim*2);
    #pragma omp declare reduction(+ : Eigen::MatrixXd : omp_out = omp_out + omp_in) \
        initializer(omp_priv = Eigen::MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))
    // 平均ヤコビ行列の計算
    #pragma omp parallel for num_threads(numThreads) firstprivate(SM, numTimeSteps) reduction(+:average_jacobian)
    for (int i = 0; i < numTimeSteps; ++i) {
        // ヤコビアンの計算
        Eigen::MatrixXd jacobian = SM.jacobian_matrix(Data.col(i));
        average_jacobian += jacobian / numTimeSteps;
    }

    Eigen::MatrixXd average(dim+1, SM.steps + 1);

    #pragma omp paralell for num_threads(numThreads) firstprivate(SM, average_jacobian, r, repetitions, dim) reduction(mat_add:average)
    for (int h; h<repetitions; h++){
        if(omp_get_thread_num == 0){
            std::cout << "processing..." << h*numThreads << "/" << repetitions << std::flush;
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> s(-M_PI, M_PI);

        Eigen::VectorXcd x_0 = Eigen::VectorXd::Zero(dim+1);
        double theta = s(gen);
        
        x_0(error_add_dim - 1) += std::complex<double>(r*std::cos(theta), r*std::sin(theta));

        Eigen::MatrixXcd traj(dim+1, SM.steps + 1);
        traj.col(0) = x_0;
        
        // ヤコビアンを作用させる用の実ベクトル
        Eigen::VectorXd state = Eigen::VectorXd::Zero(dim*2);
        for (int i = 0; i < dim; ++i) {
            state(2*i) = x_0(i).real();
            state(2*i+1) = x_0(i).imag();
        }
        double now = 0;
        // ヤコビ行列による時間発展
        for (int i = 1; i < traj.cols(); i++){
            now += dt;
            state = myfunc::rungeKuttaJacobian(state, average_jacobian, dt);
            for (int j = 0; j < dim; j++){
                std::complex<double> tmp(state(2*j), state(2*j+1));
                traj(j, i) = tmp;
            }
            traj(dim, i) = now;
        }
        average += traj.cwiseAbs() / repetitions;
    }
    
    // 結果の表示
    std::cout << "plotting" << std::endl;
    std::cout << average.rows() << std::endl;
    std::cout << dim << std::endl;
    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(800, 3000);
    int skip = 1; // plot every skip points
    std::vector<double> x((average.cols()-1)/skip),y((average.cols()-1)/skip);
    //time
    for(int i=0;i<x.size();i++){
        x[i]=Time(Time.rows()-1, i*skip);
    }
    //plot
    for(int i=0; i < dim; i++){
        for(int j=0; j < y.size(); j++){
            y[j]=average(i, j*skip);
        }
        plt::subplot(dim,1, i+1);
        plt::yscale("log");
        plt::plot(x,y);
        plt::xlabel("Time");
        plt::ylabel("$U_{" + std::to_string(i+1) + "}$");
    }
    std::map<std::string, double> keywords;
    keywords.insert(std::make_pair("hspace", 0.5)); // also right, top, bottom
    keywords.insert(std::make_pair("wspace", 0.5)); // also hspace
    plt::subplots_adjust(keywords);

    std::ostringstream oss;
    oss << "../../traj_img/jacobian_beta_" << params.beta << "nu_" << params.nu <<"_"<< t-t_0 << "period"<<repetitions << "repeat.png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    oss.str("");
     // 文字列を取得する
    oss << "../../beta" << params.beta << "_nu" << params.nu <<"_"<< t-t_0 << "period.npy";  // 文字列を結合する
    std::string npyfname = oss.str();
    // std::cout << "Saving result to " << npyfname << std::endl;
    // EigenMt2npy(traj, npyfname);

    myfunc::duration(start); // 計測終了時間
}