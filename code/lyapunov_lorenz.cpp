#include "cnpy/cnpy.h"
#include <iostream>
#include <vector>
#include <complex>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

Eigen::MatrixXd npy2EigenMat(const char* fname);
using namespace Eigen;

// 関数プロトタイプ
MatrixXd computeJacobian(const VectorXd& state, double sigma, double rho, double beta);
VectorXd computeLyapunovExponents(const MatrixXd& jacobian, int numIterations);
VectorXd rungeKuttaStep(const VectorXd& state, double dt, double sigma, double rho, double beta);
VectorXd computeLorenzDerivative(const VectorXd& state, double sigma, double rho, double beta);

// メイン関数
int main() {
    // データの読み込みをここに記述
    Eigen::MatrixXd rawData = npy2EigenMat("../../lorenz.npy");
    // パラメータの設定（例）
    int dim = 3;
    double sigma = 10.0;
    double rho = 28.0;
    double beta = 8.0 / 3.0;

    int numTimeSteps = rawData.cols();
    int numVariables = rawData.rows();

    // ヤコビアンの計算
    MatrixXd jacobian = computeJacobian(rawData.col(0), sigma, rho, beta);
    HouseholderQR<MatrixXd> qr(jacobian);
    MatrixXd Q = qr.householderQ();
    MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();  // 行列Rを取得
    VectorXd sum = R.diagonal();

    for (int i = 1; i < numTimeSteps; ++i) {
        VectorXd state = rawData.col(i);
        // QR分解によるヤコビアンの更新
        jacobian = computeJacobian(state, sigma, rho, beta);
        MatrixXd jacobi_Q = jacobian * Q;
        HouseholderQR<MatrixXd> qr(jacobi_Q);
        Q = qr.householderQ();
        R = qr.matrixQR().triangularView<Eigen::Upper>();
        sum += R.diagonal();
    }
    VectorXd lyapunovExponents = sum.array() / numTimeSteps;

    // 結果の表示
    std::cout << lyapunovExponents.rows() << std::endl;
    std::cout << "Lyapunov Exponents:" << std::endl;
    std::cout << lyapunovExponents << std::endl;

    return 0;
}

// ヤコビアンの計算
MatrixXd computeJacobian(const VectorXd& state, double sigma, double rho, double beta){
    int dim = state.rows();
    MatrixXd jacobian(dim, dim);
    double dt = 0.01;

    for (int i = 0; i < dim; ++i) {
        VectorXd perturbedState = state; // 毎回初期化
        perturbedState(i) += 1e-8;  // 無限小の微小変化をi変数のみに加える

        // 微分方程式での時間発展(1ステップ)
        Vector3d ddt_u_perturbed;
        ddt_u_perturbed = computeLorenzDerivative(perturbedState, sigma, rho, beta);
        jacobian.col(i) = (ddt_u_perturbed - perturbedState) / 1e-8;
    }
    // jacobian.row(0) = Vector3d(-sigma, sigma, 0.0);
    // jacobian.row(1) = Vector3d(rho - state(2), -1.0, -state(0));
    // jacobian.row(2) = Vector3d(state(1), state(0), -beta);

    return jacobian;
}

// ルンゲ＝クッタ法による1ステップの更新
VectorXd rungeKuttaStep(const VectorXd& state, double dt, double sigma, double rho, double beta) {
    VectorXd k1, k2, k3, k4;
    VectorXd nextState;

    k1 = dt * computeLorenzDerivative(state, sigma, rho, beta);
    k2 = dt * computeLorenzDerivative(state + 0.5 * k1, sigma, rho, beta);
    k3 = dt * computeLorenzDerivative(state + 0.5 * k2, sigma, rho, beta);
    k4 = dt * computeLorenzDerivative(state + k3, sigma, rho, beta);

    nextState = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

    return nextState;
}

// ローレンツ方程式の右辺を計算する関数
VectorXd computeLorenzDerivative(const VectorXd& state, double sigma, double rho, double beta) {
    VectorXd derivative(3);

    derivative(0) = sigma * (state(1) - state(0));
    derivative(1) = state(0) * (rho - state(2)) - state(1);
    derivative(2) = state(0) * state(1) - beta * state(2);

    return derivative;
}

Eigen::MatrixXd npy2EigenMat(const char* fname) {
    std::string fname_str(fname);
    cnpy::NpyArray arr = cnpy::npy_load(fname_str);
    if (arr.word_size != sizeof(double)) {
        throw std::runtime_error("Unsupported data type in the npy file.");
    }
    Eigen::Map<const Eigen::MatrixXd> MatT(arr.data<double>(), arr.shape[1], arr.shape[0]);
    return MatT.transpose();
}
//hit 0622

