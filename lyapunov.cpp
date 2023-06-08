#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "cnpy/cnpy.h"

Eigen::MatrixXcd npy2EigenMat(const char* fname);
void computeLyapunovExponents(const Eigen::MatrixXcd& trajectory, Eigen::VectorXd& lyapunovExponents);

int main() {
    // 軌道データの設定（Eigen::MatrixXcdを使用）
    Eigen::MatrixXcd trajectory = npy2EigenMat("../generated_laminar_beta_0.418nu_0.000173_20000period1500check400progresseps0.1.npy");
    // trajectoryの設定...
    std::cout << trajectory.rows() << trajectory.cols() << std::endl;
    int numDimensions = trajectory.rows() - 1;
    Eigen::VectorXd lyapunovExponents(numDimensions);
    lyapunovExponents.setZero();

    // リアプノフ指数の計算
    computeLyapunovExponents(trajectory.topRows(trajectory.rows() - 1), lyapunovExponents);

    // 結果の出力
    for (int i = 0; i < numDimensions; i++) {
        std::cout << "Lyapunov Exponent " << i << ": " << lyapunovExponents(i) << std::endl;
    }

    return 0;
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

void computeLyapunovExponents(const Eigen::MatrixXcd& trajectory, Eigen::VectorXd& lyapunovExponents) {
    int n = trajectory.rows();
    int m = trajectory.cols();
    int numIterations = m - 1;  // イテレーション数は軌道データの数-1

    // 初期条件の微小摂動
    Eigen::MatrixXcd v(n, n);
    v.setIdentity();

    // ヤコビアン行列の初期化
    Eigen::MatrixXcd J(n, n);
    J.setZero();

    // リアプノフ指数の計算
    for (int i = 0; i < numIterations; i++) {
        // 現在の軌道点
        Eigen::MatrixXcd x = trajectory.col(i);

        // 次の軌道点
        Eigen::MatrixXcd xNext = trajectory.col(i + 1);

        // ヤコビアン行列の計算
        J = (xNext - x) * v;
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // リアプノフ指数の更新
        for (int j = 0; j < n; j++) {
            lyapunovExponents(j) += std::log(svd.singularValues()(j));
        }

        // ヤコビアン行列の正規化
        v = svd.matrixU() * svd.matrixV().adjoint();
    }

    // リアプノフ指数の平均化
    lyapunovExponents /= static_cast<double>(numIterations);
}