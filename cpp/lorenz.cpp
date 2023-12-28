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
#include <iostream>
#include <eigen3/Eigen/Dense>
#include "cnpy/cnpy.h"

using namespace Eigen;

// ローレンツ方程式の右辺関数
Vector3d lorenz(const Vector3d& x, double sigma, double rho, double beta) {
    Vector3d dxdt;
    dxdt(0) = sigma * (x(1) - x(0));
    dxdt(1) = x(0) * (rho - x(2)) - x(1);
    dxdt(2) = x(0) * x(1) - beta * x(2);
    return dxdt;
}

// 4段4次ルンゲクッタ法
void rungeKutta4(Vector3d& x, double sigma, double rho, double beta, double dt) {
    Vector3d k1, k2, k3, k4;

    k1 = lorenz(x, sigma, rho, beta);
    k2 = lorenz(x + dt / 2.0 * k1, sigma, rho, beta);
    k3 = lorenz(x + dt / 2.0 * k2, sigma, rho, beta);
    k4 = lorenz(x + dt * k3, sigma, rho, beta);

    x += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

int main() {
    // 初期条件とパラメータ
    double sigma = 10.0;
    double rho = 28.0;
    double beta = 8.0 / 3.0;
    double dt = 0.001;
    int time = 1e+4;
    int steps = static_cast<int>(time / dt + 0.5);
    Eigen::MatrixXd Mat = Eigen::MatrixXd::Zero(3, steps + 1);

    // 初期状態
    Vector3d x(1.0, 1.0, 1.0);
    
    Mat.col(0) = x;
    // シミュレーションループ
    for (int i = 1; i < steps+1; ++i) {
        rungeKutta4(x, sigma, rho, beta, dt);
        Mat.col(i) = x;
    }

