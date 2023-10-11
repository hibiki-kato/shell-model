/**
 * @file random_jacobian.cpp
 * @author Hibiki Kato
 * @brief calc using randomly chosen jacobian matrix
 * @version 0.1
 * @date 2023-09-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <math.h>
#include <random>
#include <chrono>
#include <numeric>
#include <omp.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "Runge_Kutta.hpp"
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

Eigen::MatrixXcd npy2EigenMat(const char* fname);
Eigen::VectorXcd npy2EigenVec(const char* fname);
using namespace Eigen;

// 関数プロトタイプ
MatrixXd computeJacobian(const VectorXd& state, Eigen::VectorXd k_n, double beta, double nu);
VectorXd computeDerivativeJacobian(const VectorXd& state, const MatrixXd& jacobian);
VectorXd rungeKuttaJacobian(const VectorXd& state, const MatrixXd& jacobian, double dt);

// メイン関数
int main() {
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double nu = 0.00001;
    double beta = 0.5;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 400;
    double latter = 1;
    int threads = omp_get_max_threads();
    Eigen::VectorXcd dummy = Eigen::VectorXd::Zero(15);
    
    double repetitions = 1;
    double r = 1E-5;
    ShellModel SM(nu, beta, f, dt, t_0, t, latter, dummy);
    // データは読み込み必須
    Eigen::MatrixXcd rawData = npy2EigenMat("../../beta0.5_nu1e-05_100000period.npy");
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
    
    // 整数の一葉乱数を作成する
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_int_distribution<> dist(0, numTimeSteps-1);

    int candidates = 10;
    // ヤコビ行列をcandidates個横に並べたワイドな行列
    Eigen::MatrixXd jacobian_matrix(numVariables, numVariables * candidates);
    // 平均ヤコビ行列の計算
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < candidates; ++i) {
        VectorXd state = Data.col(dist(engine)); // ランダムにデータを選ぶ
        // ヤコビアンの計算
        Eigen::MatrixXd jacobian = computeJacobian(state, SM.get_k_n_(), SM.get_beta_(), SM.get_nu_());
        // #pragma omp critical
        jacobian_matrix.middleCols(i*numVariables, numVariables) = jacobian;
    }
    

    Eigen::MatrixXd average(dim+1, SM.get_steps_() + 1);

    #pragma omp paralell for num_threads(threads)
    for (int h; h<repetitions; h++){
        if(omp_get_thread_num == 0){
            std::cout << "\r processing..." << h*threads << "/" << repetitions << std::flush;
        }
        std::cout << "\r processing..." << h*threads << "/" << repetitions << std::flush;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> s(-M_PI, M_PI);

        Eigen::VectorXcd x_0 = Eigen::VectorXd::Zero(16);
        double theta = s(gen);
        
        x_0(12) += std::complex<double>(r*std::cos(theta), r*std::sin(theta));

        MatrixXcd traj(dim+1, SM.get_steps_() + 1);
        traj.col(0) = x_0;
        
        // ヤコビアンを作用させる用の実ベクトル
        VectorXd state = VectorXd::Zero(dim*2);
        for (int i = 0; i < dim; ++i) {
            state(2*i) = x_0(i).real();
            state(2*i+1) = x_0(i).imag();
        }
        double now = 0;
        // ヤコビ行列による時間発展
        // ヤコビ行列の選択
        std::uniform_int_distribution<> dist(0, candidates-1);
        // Eigen::MatrixXd jacobian = jacobian_matrix.middleCols(dist(engine)*numVariables, numVariables); // 1つのjacobi行列をランダムに選択し時間発展
        for (int i = 1; i < traj.cols(); i++){
            now += dt;
            Eigen::MatrixXd jacobian = jacobian_matrix.middleCols(dist(engine)*numVariables, numVariables); // ステップごとの
            state = rungeKuttaJacobian(state, jacobian, dt);
            for (int j = 0; j < dim; j++){
                std::complex<double> tmp(state(2*j), state(2*j+1));
                traj(j, i) = tmp;
            }
            traj(dim, i) = now;
        }
        #pragma omp critical
        average += traj / repetitions;

    }

    
    /*
     ██     █                    █                        █     █  █       ██   █               █  █
    ██████  █                    █                       ██     █  █     █████  █               █  █
    ██   █  █          ██   ██                           ███    █  █     █      █               █  █
    ██   █  █   ████  ████ ████  █  ██████   █████      ██ █    █  █     █      ██████   ████   █  █  █████
    ██   █  █  ██   █  █    █    █  ██   █  ██  ██      █  █    █  █     ███    ██   █  █   ██  █  █  █
    █████   █  █    █  █    █    █  █    █  █    █      █  ██   █  █       ███  █    █  █   ██  █  █  ██
    ██      █  █    █  █    █    █  █    █  █    █     ██████   █  █         ██ █    █  ██████  █  █   ███
    ██      █  █    █  █    █    █  █    █  █    █     █    ██  █  █         ██ █    █  █       █  █     ██
    ██      █  ██  ██  ██   ██   █  █    █  ██  ██     █     █  █  █         █  █    █  ██      █  █     ██
    ██      █   ████    ███  ███ █  █    █   █████    █      █  █  █     █████  █    █   ████   █  █  ████
                                                 █
                                                ██
                                            █████
    */
    //結果の表示
    std::cout << "plotting" << std::endl;
    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(2000, 3000);
    int skip = 1; // plot every skip points
    std::vector<double> x((average.cols()-1)/skip),y((average.cols()-1)/skip);
    //time
    for(int i=0;i<x.size();i++){
        x[i]=Time(i*skip);
    }
    //plot
    for(int i=0; i < dim; i+=1){
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
    oss << "../../traj_images/jacobian_beta_" << beta << "nu_" << nu <<"_"<< t-t_0 << "period"<<repetitions << "repeat.png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);
    
    /*
     ██     █                    █                      ███                          █
    ██████  █                    █                     ██████                        █
    ██   █  █          ██   ██                         ██    █                                                 ██
    ██   █  █   ████  ████ ████  █  ██████   █████     ██    ██   ████   █████ ████  █  ██████   ████  ██████ ████
    ██   █  █  ██   █  █    █    █  ██   █  ██  ██     ██     █  ██   █  ██  ██   █  █  ██   █      █  ██   █  █
    █████   █  █    █  █    █    █  █    █  █    █     ██     █  █    █  █    █   █  █  █    █      █  █    █  █
    ██      █  █    █  █    █    █  █    █  █    █     ██    ██  █    █  █    █   █  █  █    █  █████  █    █  █
    ██      █  █    █  █    █    █  █    █  █    █     ██    ██  █    █  █    █   █  █  █    █  █   █  █    █  █
    ██      █  ██  ██  ██   ██   █  █    █  ██  ██     ██   ██   ██  ██  █    █   █  █  █    █  █   █  █    █  ██
    ██      █   ████    ███  ███ █  █    █   █████     ██████     ████   █    █   █  █  █    █  █████  █    █   ███
                                                 █
                                                ██
                                            █████
    */
    // // calculate error ratio of each shell
    // for (int i = 0; i < average.cols(); i++) {
    //     average.block(0, i, dim, 1) /= average.block(0, i, dim, 1).sum();
    // }
    // // 結果の表示
    // std::cout << "plotting" << std::endl;
    // // plot settings
    // std::map<std::string, std::string> plotSettings;
    // plotSettings["font.family"] = "Times New Roman";
    // plotSettings["font.size"] = "10";
    // plt::rcparams(plotSettings);
    // // Set the size of output image = 1200x780 pixels
    // plt::figure_size(1200, 780);
    // int skip = 100; // plot every skip points
    // std::vector<double> x((average.cols()-1)/skip),y((average.cols()-1)/skip);
    // //time
    // for(int i=0;i<x.size();i++){
    //     x[i]=Time(i*skip);
    // }
    // //plot
    // for(int i=0; i < dim; i++){
    //     if (i == 0 | i == 1 | i == 3 | i == 4 | i == 7 | i == 8){
    //         for(int j=0; j < y.size(); j++){
    //             y[j]=average(i, j*skip);
    //         }
    //         std::map<std::string, std::string> keywords;
    //         keywords.insert(std::pair<std::string, std::string>("label", std::to_string(i+1)+"th shell"));
    //         plt::plot(x, y, keywords);
    //     }
    // }
    // plt::xscale("log");
    // plt::yscale("log");
    // plt::xlabel("Time");
    // plt::ylabel("Ratio among shells");
    // plt::legend();

    // std::ostringstream oss;
    // oss << "../../error_dominant_shell/jacobian_beta_" << beta << "nu_" << nu <<"_"<< t-t_0 << "period"<<repetitions << "repeat.png";  // 文字列を結合する
    // std::string plotfname = oss.str(); // 文字列を取得する
    // std::cout << "Saving result to " << plotfname << std::endl;
    // plt::save(plotfname);

    auto end = std::chrono::system_clock::now();  // 計測終了時間
    int hours = std::chrono::duration_cast<std::chrono::hours>(end-start).count(); //処理に要した時間を変換
    int minutes = std::chrono::duration_cast<std::chrono::minutes>(end-start).count(); //処理に要した時間を変換
    int seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count(); //処理に要した時間を変換
    int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間を変換
    std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;
    return 0;
}

// ヤコビアンの計算
MatrixXd computeJacobian(const VectorXd& state, Eigen::VectorXd k_n, double beta, double nu){
    int dim = state.rows();
    MatrixXd jacobian = Eigen::MatrixXd::Zero(dim, dim);
    
    // A
    for (int i = 0; i < dim/2 - 2; ++i) {
        jacobian(2*i, 2*i + 2) += k_n(i) * state((i+2)*2 + 1);
        jacobian(2*i, 2*i+1 + 2) += k_n(i) * state((i+2)*2);
        jacobian(2*i+1, 2*i + 2) += k_n(i) * state((i+2)*2);
        jacobian(2*i+1, 2*i+1 + 2) += -k_n(i) * state((i+2)*2 + 1);

        jacobian(2*i, 2*i + 4) +=  k_n(i) * state((i+1)*2 + 1);
        jacobian(2*i, 2*i+1 + 4) += k_n(i) * state((i+1)*2);
        jacobian(2*i+1, 2*i + 4) +=  k_n(i) * state((i+1)*2);
        jacobian(2*i+1, 2*i+1 + 4) += -k_n(i) * state((i+1)*2 + 1);
    }

    // B
    for (int i = 1; i < dim/2 - 1; ++i) {
        jacobian(2*i, 2*i - 2) +=  -beta * k_n(i-1) * state((i+1)*2 + 1);
        jacobian(2*i, 2*i+1 - 2) += -beta * k_n(i-1) * state((i+1)*2);
        jacobian(2*i+1, 2*i - 2) += -beta * k_n(i-1) * state((i+1)*2);
        jacobian(2*i+1, 2*i+1 - 2) += beta * k_n(i-1) * state((i+1)*2 + 1);

        jacobian(2*i, 2*i + 2) +=  -beta * k_n(i-1) * state((i-1)*2 + 1);
        jacobian(2*i, 2*i+1 + 2) += -beta * k_n(i-1) * state((i-1)*2);
        jacobian(2*i+1, 2*i + 2) +=  -beta * k_n(i-1) * state((i-1)*2);
        jacobian(2*i+1, 2*i+1 + 2) += beta * k_n(i-1) * state((i-1)*2 + 1);
    }

    // Gamma
    for (int i = 2; i < dim/2; ++i) {
        jacobian(2*i, 2*i - 4) +=  (beta-1) * k_n(i-2) * state((i-1)*2 + 1);
        jacobian(2*i, 2*i+1 - 4) += (beta-1) * k_n(i-2) * state((i-1)*2);
        jacobian(2*i+1, 2*i - 4) += (beta-1) * k_n(i-2) * state((i-1)*2);
        jacobian(2*i+1, 2*i+1 - 4) += (1-beta) * k_n(i-2) * state((i-1)*2 + 1);

        jacobian(2*i, 2*i - 2) +=  (beta-1) * k_n(i-2) * state((i-2)*2 + 1);
        jacobian(2*i, 2*i+1 - 2) += (beta-1) * k_n(i-2) * state((i-2)*2);
        jacobian(2*i+1, 2*i - 2) +=  (beta-1) * k_n(i-2) * state((i-2)*2);
        jacobian(2*i+1, 2*i+1 - 2) += (1-beta) * k_n(i-2) * state((i-2)*2 + 1);
    }
    
    // N
    for (int i = 0; i < dim/2; ++i) {
        jacobian(2*i, 2*i) = -nu*k_n(i)*k_n(i);
        jacobian(2*i+1, 2*i+1) = -nu*k_n(i)*k_n(i);
    }

    return jacobian;
}

// ルンゲ＝クッタ法を用いた"ヤコビアン"による時間発展
VectorXd rungeKuttaJacobian(const VectorXd& state, const MatrixXd& jacobian, double dt){
    VectorXd k1, k2, k3, k4;
    VectorXd nextState;
    
    k1 = dt * computeDerivativeJacobian(state, jacobian);
    k2 = dt * computeDerivativeJacobian(state + 0.5 * k1, jacobian);
    k3 = dt * computeDerivativeJacobian(state + 0.5 * k2, jacobian);
    k4 = dt * computeDerivativeJacobian(state + k3, jacobian);

    nextState = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

    return nextState;
}

VectorXd computeDerivativeJacobian(const VectorXd& state, const MatrixXd& jacobian) {
    VectorXd derivative(state.rows());
    // VectorXd energy_scale = VectorXd::Zero(state.rows());
    // std::vector<double> energy = {0.287058,
    //                             0.192392,
    //                             0.206979,
    //                             0.101921,
    //                             0.0960957,
    //                             0.0836411,
    //                             0.0567056,
    //                             0.0414303,
    //                             0.0369566,
    //                             0.0252059,
    //                             0.0153319,
    //                             0.0151464,
    //                             0.00973406,
    //                             0.00322299,
    //                             0.000500837};
    // for(int i=0; i < energy.size(); i++){
    //     energy_scale(2*i) = energy[i];
    //     energy_scale(2*i+1) = energy[i];
    // }
    // derivative = jacobian * (state.array() * (1-(state.array()/energy_scale.array()))).matrix();
    // //derivativeにnanが含まれていたらプログラムを停止
    // if(derivative.hasNaN()){
    //     std::cout << "derivative has NaN" << std::endl;
    //     std::cout << state << std::endl;
    //     std::cout << energy_scale << std::endl;
    //     std::cout << (1-abs(state.array()/energy_scale.array())) << std::endl;
    //     exit(1);
    // }
    // std::cout << state << std::endl;
    // std::cout << "罰則高は" << std::endl;
    // std::cout << (1-abs(state.array()/energy_scale.array())) << std::endl;
    return derivative;
}

Eigen::MatrixXcd npy2EigenMat(const char* fname){
    std::string fname_str(fname);
    cnpy::NpyArray arr = cnpy::npy_load(fname_str);
    if (arr.word_size != sizeof(std::complex<double>)) {
        throw std::runtime_error("Unsupported data type in the npy file.");
    }
    Eigen::Map<const Eigen::MatrixXcd> MatT(arr.data<std::complex<double>>(), arr.shape[1], arr.shape[0]);
    return MatT.transpose();
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