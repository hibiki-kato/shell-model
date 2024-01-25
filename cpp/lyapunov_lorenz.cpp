#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <chrono>
#include <string>
#include <numeric>
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"

Eigen::MatrixXd npy2EigenMat(const char* fname);

// メイン関数
int main() {
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    // パラメータの設定（例）
    int dim = 3;
    Lorenz63params params;
    params.sigma = 10.0;
    params.rho = 28.0;
    params.beta = 8.0 / 3.0;
    double dt = 0.01;
    int t_0 = 0;
    int t = 1e+5;
    int dump = 1e+4;
    // 初期値の設定
    Eigen::VectorXd x_0 = Eigen::VectorXd::Random(dim);

    Lorenz63 L63(params, dt, t_0, t, dump, x_0);
    Eigen::MatrixXd trajectory = L63.get_trajectory().topRows(dim);
    Eigen::VectorXd lyapunovExponents = myfunc::calcLyapunovExponent(L63, trajectory, 1);

    // 結果の表示
    std::cout << lyapunovExponents.rows() << std::endl;
    std::cout << "Lyapunov Exponents:" << std::endl;
    std::cout << lyapunovExponents << std::endl;

    myfunc::duration(start);
    return 0;
}