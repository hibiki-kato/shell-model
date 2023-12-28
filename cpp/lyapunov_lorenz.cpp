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
#include "cnpy/cnpy.h"
#include <iostream>
#include <vector>
#include <complex>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

Eigen::MatrixXd npy2EigenMat(const char* fname);
using namespace Eigen;

// 関数プロトタイプ
MatrixXd computeJacobian(const VectorXd& state, double sigma, double rho, double beta, double dt = 0.01);
VectorXd rungeKuttaJacobian(const VectorXd& state, const MatrixXd& jacobian, double dt);
VectorXd computeDerivativeJacobian(const VectorXd& state, const MatrixXd& jacobian);
VectorXd rungeKuttaStep(const VectorXd& state, double sigma, double rho, double beta, double dt);
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

    double dt = 0.01;

    int numTimeSteps = rawData.cols();
    int numVariables = rawData.rows();
    //任意の直行行列を用意する
    MatrixXd Base = Eigen::MatrixXd::Random(numVariables, numVariables);
    HouseholderQR<MatrixXd> qr(Base);
    Base = qr.householderQ();
    // 総和の初期化
    VectorXd sum = Eigen::VectorXd::Zero(numVariables);
    // 次のステップ(QR分解されるもの)
    MatrixXd next(numVariables, numVariables);

    for (int i = 0; i < numTimeSteps; ++i) {
        VectorXd state = rawData.col(i);
        // ヤコビアンの計算
        auto jacobian = computeJacobian(state, sigma, rho, beta, dt);
        // ヤコビアンとBase(直行行列)の積を計算する
        for (int j = 0; j < numVariables; ++j) {
            next.col(j) = rungeKuttaJacobian(Base.col(j), jacobian, dt);
        }

        // QR分解を行う
        HouseholderQR<MatrixXd> qr(next);
        // 直交行列QでBaseを更新
        Base = qr.householderQ();
        // Rの対角成分を総和に加える
        Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
        // Rの対角成分の絶対値のlogをsumにたす
        Eigen::VectorXd diag = R.diagonal().cwiseAbs().array().log();
        sum += diag;
    }
    std::cout << numTimeSteps << std::endl;

    VectorXd lyapunovExponents = sum.array() / numTimeSteps;

    // 結果の表示
    std::cout << lyapunovExponents.rows() << std::endl;
    std::cout << "Lyapunov Exponents:" << std::endl;
    std::cout << lyapunovExponents << std::endl;

    return 0;
}

// ヤコビアンの計算
MatrixXd computeJacobian(const VectorXd& state, double sigma, double rho, double beta, double dt){
    int dim = state.rows();
    MatrixXd jacobian(dim, dim);
    // for (int i = 0; i < dim; ++i) {
    //     VectorXd perturbedState = state; // 毎回初期化
    //     perturbedState(i) += 1e-7;  // 無限小の微小変化をi変数のみに加える

    //     // 微分方程式での時間発展(1ステップ)
    //     Vector3d ddt_u_perturbed;
    //     ddt_u_perturbed = rungeKuttaStep(perturbedState, sigma, rho, beta, dt);
    //     jacobian.col(i) = (ddt_u_perturbed - perturbedState) / 1e-7;
    // }
    jacobian.row(0) = Vector3d(-sigma, sigma, 0.0);
    jacobian.row(1) = Vector3d(rho - state(2), -1.0, -state(0));
    jacobian.row(2) = Vector3d(state(1), state(0), -beta);

    return jacobian;
}

// ルンゲ＝クッタ法を用いたヤコビアンによる時間発展
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
// ルンゲクッタ法を用いた時間発展
VectorXd rungeKuttaStep(const VectorXd& state, double sigma, double rho, double beta, double dt){
    VectorXd k1, k2, k3, k4;
    VectorXd nextState;

    k1 = dt * computeLorenzDerivative(state, sigma, rho, beta);
    k2 = dt * computeLorenzDerivative(state + 0.5 * k1, sigma, rho, beta);
    k3 = dt * computeLorenzDerivative(state + 0.5 * k2, sigma, rho, beta);
    k4 = dt * computeLorenzDerivative(state + k3, sigma, rho, beta);

    nextState = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

    return nextState;
}

VectorXd computeDerivativeJacobian(const VectorXd& state, const MatrixXd& jacobian) {
    VectorXd derivative(3);
    derivative = jacobian * state;
    return derivative;
}

VectorXd computeLorenzDerivative(const VectorXd& state, double sigma, double rho, double beta){
    VectorXd derivative(3);
    derivative(0) = sigma * (state(1) - state(0));
    derivative(1) = rho * state(0) - state(1) - state(0) * state(2);
    derivative(2) = state(0) * state(1) - beta * state(2);
    return derivative;
}

Eigen::MatrixXd npy2EigenMat(const char* fname){
    std::string fname_str(fname);
    cnpy::NpyArray arr = cnpy::npy_load(fname_str);
    if (arr.word_size != sizeof(double)) {
        throw std::runtime_error("Unsupported data type in the npy file.");
    }
    Eigen::Map<const Eigen::MatrixXd> MatT(arr.data<double>(), arr.shape[1], arr.shape[0]);
    return MatT.transpose();
}
//hit 0622

