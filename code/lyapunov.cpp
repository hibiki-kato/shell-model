#include "cnpy/cnpy.h"
#include <iostream>
#include <vector>
#include <complex>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

Eigen::MatrixXcd npy2EigenMat(const char* fname);
using namespace Eigen;

// 関数プロトタイプ
MatrixXcd computeJacobian(const VectorXcd& state, const VectorXd& c_n_1, const VectorXd& c_n_2, const VectorXd& c_n_3, const VectorXd& k_n, double nu, std::complex<double> f);
VectorXcd computeLyapunovExponents(const MatrixXcd& jacobian, int numIterations);

// メイン関数
int main() {
    // データの読み込みをここに記述
    Eigen::MatrixXcd loaded = npy2EigenMat("../../generated_lam/generated_laminar_beta_0.42nu_0.00018_131400period1400check400progresseps0.1.npy");
    Eigen::MatrixXcd rawData = loaded.topRows(14);
    // パラメータの設定（例）
    int dim = 14;
    double nu = 0.00018;
    double beta = 0.417;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double q = 2.0;
    double k_0 = pow(2, -4);
    Eigen::VectorXd k_n = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd c_n_1 = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd c_n_2 = Eigen::VectorXd::Zero(dim);
    Eigen::VectorXd c_n_3 = Eigen::VectorXd::Zero(dim);
    for (int i = 0; i < dim; i++) {
        k_n(i) = k_0 * pow(q, i+1);
    };
    c_n_1 = Eigen::VectorXd::Zero(dim);
    c_n_1.topRows(dim-2) = k_n.topRows(dim-2);

    c_n_2 = Eigen::VectorXd::Zero(dim);
    c_n_2.middleRows(1, dim-2) = k_n.topRows(dim-2).array() * (-beta);

    c_n_3 = Eigen::VectorXd::Zero(dim);
    c_n_3.bottomRows(dim-2) = k_n.topRows(dim-2).array() * (beta - 1);

    int numTimeSteps = rawData.rows();
    int numVariables = rawData.cols();

    // ヤコビアンの計算
    MatrixXcd jacobian = computeJacobian(rawData.col(0), c_n_1, c_n_2, c_n_3, k_n, nu, f);

    for (int i = 1; i < numTimeSteps; ++i) {
        VectorXcd state = rawData.col(i).transpose();

        // QR分解によるヤコビアンの更新
        MatrixXcd jacobianUpdate = computeJacobian(state, c_n_1, c_n_2, c_n_3, k_n, nu, f);
        HouseholderQR<MatrixXcd> qr(jacobian);
        jacobian = qr.householderQ() * jacobianUpdate;
    }

    // リアプノフ指数の計算
    VectorXcd lyapunovExponents = computeLyapunovExponents(jacobian, numVariables);

    // 結果の表示
    std::cout << "Lyapunov Exponents:" << std::endl;
    std::cout << lyapunovExponents << std::endl;

    return 0;
}

// ヤコビアンの計算
MatrixXcd computeJacobian(const VectorXcd& state, const VectorXd& c_n_1, const VectorXd& c_n_2, const VectorXd& c_n_3, const VectorXd& k_n, double nu, std::complex<double> f) {
    int dim = state.rows();
    VectorXcd u = VectorXcd::Zero(dim + 4);

    MatrixXcd jacobian(dim, dim);

    for (int i = 0; i < dim; ++i) {
        VectorXcd perturbedState = state; // 毎回初期化
        perturbedState(i) += std::complex<double>(1e-10, 0);  // 無限小の微小変化をi変数のみに加える
        u.segment(2, dim) = perturbedState;

        // 微分方程式での時間発展(1ステップ)
        VectorXcd ddt_u_perturbed = (c_n_1.array() * u.segment(3, dim).conjugate().array() * u.tail(dim).conjugate().array() +
                                     c_n_2.array() * u.segment(1, dim).conjugate().array() * u.segment(3, dim).conjugate().array() +
                                     c_n_3.array() * u.segment(1, dim).conjugate().array() * u.segment(0, dim).conjugate().array()) *
                                        std::complex<double>(0.0, 1.0) -
                                    nu * u.segment(2, dim).array() * k_n.array().square();
        ddt_u_perturbed(0) += f;
        // 
        jacobian.col(i) = (ddt_u_perturbed - perturbedState) / 1e-10;
    }

    return jacobian;
}

// リアプノフ指数の計算
VectorXcd computeLyapunovExponents(const MatrixXcd& jacobian, int numIterations) {
    int dim = jacobian.rows();
    VectorXcd lyapunovExponents(dim);
    VectorXcd x(dim);
    x.setRandom();
    x.normalize();

    for (int i = 0; i < numIterations; ++i) {
        x = jacobian * x;
        x.normalize();
        lyapunovExponents = lyapunovExponents + x.array().log().matrix();
    }

    lyapunovExponents = lyapunovExponents.array() / numIterations;

    return lyapunovExponents;
}
Eigen::MatrixXcd npy2EigenMat(const char* fname) {
    std::string fname_str(fname);
    cnpy::NpyArray arr = cnpy::npy_load(fname_str);
    if (arr.word_size != sizeof(std::complex<double>)) {
        throw std::runtime_error("Unsupported data type in the npy file.");
    }
    Eigen::Map<const Eigen::MatrixXcd> MatT(arr.data<std::complex<double>>(), arr.shape[1], arr.shape[0]);
    return MatT.transpose();
}
