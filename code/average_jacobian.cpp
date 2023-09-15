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
// VectorXd rungeKuttaStep(const VectorXd& state, double dt, Eigen::VectorXd c_n_1, Eigen::VectorXd c_n_2, Eigen::VectorXd c_n_3, Eigen::VectorXd k_n, double nu, std::complex<double> f);
// Eigen::VectorXcd goy_shell_model(Eigen::VectorXd state, Eigen::VectorXd c_n_1, Eigen::VectorXd c_n_2, Eigen::VectorXd c_n_3, Eigen::VectorXd k_n, double nu, std::complex<double> f);


// メイン関数
int main() {
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double nu = 0.00018;
    double beta = 0.42;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+2;
    double latter = 1;
    int threads = omp_get_max_threads();
    Eigen::VectorXcd x_0 = Eigen::Zeros(15);
    x_0(12) += std::complex<double>(1E-5, 1E-5);
    ShellModel SM(nu, beta, f, dt, t_0, t, latter, x_0);
    // データの読み込みをここに記述
    //Eigen::MatrixXcd rawData = npy2EigenMat("../generated_laminar_beta_0.419nu_0.00018_200000period1500check500progresseps0.1.npy");
    Eigen::MatrixXcd rawData = SM.get_trajectory_();
    std::cout << "trajectory has been calculated" << std::endl;
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
    //後で使う用の確保
    MatrixXcd traj(dim+1, numTimeSteps);
    traj.col(0) = rawData.col(0);

    Eigen::MatrixXd average_jacobian(dim*2, dim*2);
    // 平均ヤコビ行列の計算
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < numTimeSteps; ++i) {
        VectorXd state = Data.col(i);
        // ヤコビアンの計算
        auto jacobian = computeJacobian(state, SM.get_k_n_(), SM.get_beta_(), SM.get_nu_());
        #pragma omp critical
        average_jacobian += jacobian / numTimeSteps;
    }
    
    std::cout << "jacobian matrix was " << std::endl << average_jacobian << std::endl;

    
    // ヤコビアンを作用させる用の実ベクトル
    VectorXd state = Data.col(0);
    double now = 0;
    // ヤコビ行列による時間発展
    for (int i = 1; i < numTimeSteps; i++){
        std::cout << "\r processing..." << i << "/" << numTimeSteps << std::flush;
        now += dt;
        state = rungeKuttaJacobian(state, average_jacobian, dt);
        for (int j = 0; j < dim; j++){
            std::complex<double> tmp(state(2*j), state(2*j+1));
            traj(j, i) = tmp;
        }
        traj(dim, i) = now;
    }
    // 結果の表示
    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(400, 2000);
    // Add graph title
    std::vector<double> x(traj.cols()),y(traj.cols());
    for(int i=0;i<traj.cols();i+=100){
        x[i]=traj.cwiseAbs()(traj.rows()-1, i);
    }
    for(int i=0; i < dim; i++){
        for(int j=0; j < traj.cols(); j+=100){
            y[j]=traj.cwiseAbs()(i, j);
        }
        plt::subplot(dim,1, i+1);
        plt::scatter(x,y);
        plt::xlabel("Time");
        plt::ylabel("$U_{" + std::to_string(i+1) + "}$");
    }
    std::map<std::string, double> keywords;
    keywords.insert(std::make_pair("hspace", 0.5)); // also right, top, bottom
    keywords.insert(std::make_pair("wspace", 0.5)); // also hspace
    plt::subplots_adjust(keywords);

    std::ostringstream oss;
    oss << "../../traj_images/jacobian_beta_" << beta << "nu_" << nu <<"_"<< t-t_0 << "period.png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    oss.str("");
     // 文字列を取得する
    oss << "../../beta" << beta << "_nu" << nu <<"_"<< t-t_0 << "period.npy";  // 文字列を結合する
    std::string npyfname = oss.str();
    // std::cout << "Saving result to " << npyfname << std::endl;
    // EigenMt2npy(traj, npyfname);

    

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
    derivative = jacobian * state;
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