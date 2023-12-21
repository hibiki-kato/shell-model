/**
 * @file random_jacobian.cpp
 * @author Hibiki Kato
 * @brief calc error between a model and perturbed one, using jacobian matrix referring from original shell model.
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
#include "Eigen_numpy_converter.hpp"
namespace plt = matplotlibcpp;

// 関数プロトタイプ
Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& state, Eigen::VectorXd k_n, double beta, double nu);
Eigen::VectorXd computeDerivativeJacobian(const Eigen::VectorXd& state, const Eigen::MatrixXd& jacobian);
Eigen::VectorXd rungeKuttaJacobian(const Eigen::VectorXd& state, const Eigen::MatrixXd& jacobian, double dt);

// メイン関数
int main() {
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double nu = 1e-5;
    double beta = 0.5;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 10000;
    double latter = 1;
    int threads = omp_get_max_threads();
    Eigen::VectorXcd dummy = Eigen::VectorXd::Zero(15);
    
    int repetitions = 100;
    double r = 1E-5;
    ShellModel SM(nu, beta, f, dt, t_0, t, latter, dummy);
    // データは読み込み必須
    Eigen::MatrixXcd rawData = npy2EigenMat<std::complex<double>>("../../beta0.5_nu1e-05_10000period.npy");
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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> s(-M_PI, M_PI);
    std::atomic<double> counter = 0; // progress checker
    Eigen::MatrixXd average(dim, rawData.cols());

    
    #pragma omp parallel for num_threads(threads) schedule(dynamic) shared(counter, dim, rawData, SM, average, Data, gen, s, r, beta, nu, dt, repetitions)
    for (int h = 0; h<repetitions; h++){
        if(omp_get_thread_num == 0){
            std::cout << "\r processing..." << counter << "/" << repetitions << std::flush;
        }

        Eigen::MatrixXd Origin(dim*2, rawData.cols());
        Eigen::MatrixXd Perturbed(dim*2, rawData.cols());
        Origin.col(0) = Data.col(0);
        Perturbed.col(0) = Data.col(0);
        // perturbation
        double theta = s(gen);
        Perturbed(24, 0) += r * cos(theta);
        Perturbed(25, 0) += r * sin(theta);
        
        // time evolution
        double max_eigenvalue = 0;
        for (int i = 1; i < Data.cols(); i++){
            Eigen::MatrixXd jacobian = computeJacobian(Data.col(i-1), SM.get_k_n_(), beta, nu);
            // jacobianの最大固有値(絶対値)を計算
            // if (i == 1){
                Eigen::EigenSolver<Eigen::MatrixXd> es(jacobian);
                for (int i = 0; i < jacobian.rows(); ++i) {
                    if (max_eigenvalue < std::abs(std::real(es.eigenvalues()[i]))) {
                        max_eigenvalue = std::abs(std::real(es.eigenvalues()[i]));
                    }
                }
            // }
            jacobian /= max_eigenvalue;
            Origin.col(i) = rungeKuttaJacobian(Origin.col(i-1), jacobian, dt);
            Perturbed.col(i) = rungeKuttaJacobian(Perturbed.col(i-1), jacobian, dt);
        }
        

        // calculate error between origin and perturbed across shells
        Eigen::MatrixXd diff_square = (Origin - Perturbed).cwiseAbs2();
        Eigen::MatrixXd diff(dim, average.cols());
        for (int j = 0; j < average.cols(); j++){
            for (int i = 0; i < dim; i++){
                diff(i, j) = std::sqrt(diff_square(2*i, j) + diff_square(2*i+1, j));
            }
        }
        #pragma omp critical
        average += diff / repetitions;
        
        counter += 1;
    }

    std::ostringstream oss;
    int skip = 100; // plot every skip points
    std::vector<double> x((average.cols()-1)/skip),y((average.cols()-1)/skip);
    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    
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
    // */
    //結果の表示
    std::cout << "plotting" << std::endl;
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(2000, 3000);
    //time
    for(int i=0;i<x.size();i++){
        x[i]=i*skip*dt;
    }
    //plot
    for(int i=0; i < dim; i+=1){
        for(int j=0; j < y.size(); j++){
            y[j]=average(i, j*skip);
        }
        plt::subplot(dim,1, i+1);
        plt::yscale("log");
        plt::xscale("log");
        plt::plot(x,y);
        plt::xlabel("Time");
        plt::ylabel("$U_{" + std::to_string(i+1) + "}$");
    }
    std::map<std::string, double> keywords;
    keywords.insert(std::make_pair("hspace", 0.5)); // also right, top, bottom
    keywords.insert(std::make_pair("wspace", 0.5)); // also hspace
    plt::subplots_adjust(keywords);

    oss << "../../traj_images/jacobian_beta_" << beta << "nu_" << nu <<"_"<< t-t_0 << "period"<<repetitions << "repeat.png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);
    oss.str("");
    plt::clf();
    
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
    // calculate error ratio of each shell
    for (int i = 0; i < average.cols(); i++) {
        average.block(0, i, dim, 1) /= average.block(0, i, dim, 1).sum();
    }
    // 結果の表示
    std::cout << "plotting" << std::endl;
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1200, 780);
    //time
    for(int i=0;i<x.size();i++){
        x[i]=i*skip*dt;
    }
    //plot
    for(int i=0; i < dim; i++){
        if (i == 0 | i == 1 | i == 3 | i == 4 | i == 7 | i == 8){
            for(int j=0; j < y.size(); j++){
                y[j]=average(i, j*skip);
            }
            std::map<std::string, std::string> keywords;
            keywords.insert(std::pair<std::string, std::string>("label", std::to_string(i+1)+"th shell"));
            plt::plot(x, y, keywords);
        }
    }
    plt::xscale("log");
    // plt::yscale("log");
    plt::xlabel("Time");
    plt::ylabel("Ratio among shells");
    plt::legend();

    oss << "../../error_dominant_shell/jacobian_beta_" << beta << "nu_" << nu <<"_"<< t-t_0 << "period"<<repetitions << "repeat.png";  // 文字列を結合する
    std::string plotfname1 = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname1 << std::endl;
    plt::save(plotfname1);

    auto end = std::chrono::system_clock::now();  // 計測終了時間
    int hours = std::chrono::duration_cast<std::chrono::hours>(end-start).count(); //処理に要した時間を変換
    int minutes = std::chrono::duration_cast<std::chrono::minutes>(end-start).count(); //処理に要した時間を変換
    int seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count(); //処理に要した時間を変換
    int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間を変換
    std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;
    return 0;
}

// ヤコビアンの計算
Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& state, Eigen::VectorXd k_n, double beta, double nu){
    int dim = state.rows();
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(dim, dim);
    
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
Eigen::VectorXd rungeKuttaJacobian(const Eigen::VectorXd& state, const Eigen::MatrixXd& jacobian, double dt){
    Eigen::VectorXd k1, k2, k3, k4;
    Eigen::VectorXd nextState;
    
    k1 = dt * computeDerivativeJacobian(state, jacobian);
    k2 = dt * computeDerivativeJacobian(state + 0.5 * k1, jacobian);
    k3 = dt * computeDerivativeJacobian(state + 0.5 * k2, jacobian);
    k4 = dt * computeDerivativeJacobian(state + k3, jacobian);

    nextState = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

    return nextState;
}

Eigen::VectorXd computeDerivativeJacobian(const Eigen::VectorXd& state, const Eigen::MatrixXd& jacobian) {
    Eigen::VectorXd derivative(state.rows());
    derivative = jacobian * state;
    Eigen::VectorXd energy_scale = Eigen::VectorXd::Zero(state.rows());
    std::vector<double> energy = {6.71346879e-01, 6.41798493e-01, 5.58229424e-01, 3.73568202e-01,
       4.05778553e-01, 3.69391086e-01, 3.11250991e-01, 2.56594141e-01,
       2.15351203e-01, 1.81350920e-01, 1.75857695e-01, 1.33130565e-01,
       1.23656096e-01, 1.41755798e-01, 6.72223654e-02};
    for(int i=0; i < energy.size(); i++){
        energy_scale(2*i) = energy[i] * 1000;
        energy_scale(2*i+1) = energy[i] * 1000;
    }
    derivative = (jacobian * state).array() * (1-(state.array()/energy_scale.array()));
    // derivative = jacobian * (state.array() * (energy_scale.array()-state.array())).matrix();
    //derivativeにnanが含まれていたらプログラムを停止
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