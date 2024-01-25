/**
 * @file Flow.hpp
 * @author Hibiki Kato
 * @brief header of Flow classes
 * @version 0.1
 * @date 2023-12-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <cmath>
#include <complex>
#include <vector>
/*
██                   ██                                    ███      ████
██                   ██                                  ████      ███████
██                   ██                                 ███       ██    ██
██                   ██                                 ██        ██    ██
██          █████    ██     ████    ██ ████   ████████  █               ██
██         ███████   ██    ███████  ████████       ██  ██ ████         ███
██        ██    ███  ██   ██    ██  ██    ██      ███  ████████      ████
██        ██     ██  ██   ██     █  ██    ██      ██   ███   ██       ████
██        ██     ██  ██   ████████  ██    ██     ██    ██     ██        ██
██        ██     ██  ██   ██        ██    ██    ██     ██     ██         █
██        ██     ██  ██   ██        ██    ██   ███     ██     ██  ██     █
██        ██    ███  ██   ██        ██    ██   ██       ██   ██   ██    ██
████████   ███████   ██    ███████  ██    ██  ███       ███████    ███████
█████████   █████    ██     █████   ██    ██  ████████    ████      ████
*/

struct Lorenz63params{
    double sigma;
    double rho;
    double beta;
};

struct Lorenz63{
    Lorenz63(Lorenz63params input_params, double input_dt, double input_t_0, double input_t, double input_dump, Eigen::VectorXd input_x_0);
    ~Lorenz63();
    Eigen::MatrixXd get_trajectory();
    Eigen::VectorXd rk4(const Eigen::VectorXd& present);
    Eigen::VectorXd lorenz63(const Eigen::VectorXd& state);
    Eigen::MatrixXd jacobian_matrix(const Eigen::VectorXd& state);

    //data members
    double sigma;
    double rho;
    double beta;
    double dt;
    double t_0;
    double t;
    double dump;
    long long steps;
    long long dump_steps;
    Eigen::VectorXd x_0;
};


/*
  █████    ██                  ██   ██   ███        ███                     █             ██
 ████████  ██                  ██   ██   ████       ███                     █             ██
██     ██  ██                  ██   ██   ████       ███                     █             ██
██      ██ ██                  ██   ██   ████      ████                     █             ██
██         ██ ████     ████    ██   ██   ██ ██     ████     █████      ██████     ████    ██
 ██        ████████   ███████  ██   ██   ██ ██     █ ██    ███████    ███████    ███████  ██
  ████     ██    ██  ██    ██  ██   ██   ██  █    ██ ██   ██    ███  ██    ██   ██    ██  ██
    ████   ██    ██  ██     █  ██   ██   ██  ██   ██ ██   ██     ██  ██     █   ██     █  ██
      ███  ██    ██  ████████  ██   ██   ██  ██  ██  ██   ██     ██  ██     █   ████████  ██
       ███ ██    ██  ██        ██   ██   ██   ██ ██  ██   ██     ██  ██     █   ██        ██
██      ██ ██    ██  ██        ██   ██   ██   ██ █   ██   ██     ██  ██     █   ██        ██
██     ██  ██    ██  ██        ██   ██   ██    ███   ██   ██    ███  ██    ██   ██        ██
 ████████  ██    ██   ███████  ██   ██   ██    ███   ██    ███████    ███████    ███████  ██
  █████    ██    ██    █████   ██   ██   ██    ██    ██     █████      ████ █     █████   ██
*/

struct SMparams{
    double nu;
    double beta;
    std::complex<double> f;
};

struct ShellModel{
    ShellModel(SMparams input_prams, double input_dt, double input_t_0, double input_t, double input_dump, Eigen::VectorXcd input_x_0);
    ~ShellModel();
    Eigen::MatrixXcd get_trajectory();
    Eigen::VectorXd energy_spectrum(const Eigen::MatrixXcd& trajectory = Eigen::MatrixXcd());
    Eigen::VectorXcd rk4(const Eigen::VectorXcd& present);
    Eigen::VectorXcd goy_shell_model(const Eigen::VectorXcd& state);
    Eigen::MatrixXd jacobian_matrix(const Eigen::VectorXd& state);
    void set_beta_(double input_beta);
    void set_t_0_(double input_t_0);
    void set_t_(double input_t);

    //data members
    double nu;
    double beta;
    std::complex<double> f;
    double dt;
    double t_0;
    double t;
    double dump;
    long steps;
    long dump_steps;
    Eigen::VectorXd k_n;
    Eigen::VectorXd c_n_1;
    Eigen::VectorXd c_n_2;
    Eigen::VectorXd c_n_3;
    Eigen::VectorXcd x_0;
};

/*
   █████                                    ██                    █   ████████                                   ██
  ███████                                   ██                    █   █████████                                  ██
 ██     ██                                  ██                    █   ██     ███                                 ██
 ██     ██                                  ██                    █   ██      ██                                 ██
██       █    █████    ██    ██  ██ ████    ██     ████      ██████   ██      ██    █████      ████      ████    ██     ████    ██ ██
██           ███████   ██    ██  ████████   ██    ███████   ███████   ██      ██   ███████    ██████    ██████   ██    ███████  █████
██          ██    ███  ██    ██  ██    ██   ██   ██    ██  ██    ██   ██     ██   ██    ███  ██    ██  ██    ██  ██   ██    ██  ██
██          ██     ██  ██    ██  ██     ██  ██   ██     █  ██     █   ████████    ██     ██  ██        ██        ██   ██     █  ██
██          ██     ██  ██    ██  ██     ██  ██   ████████  ██     █   ███████     ██     ██   ████      ████     ██   ████████  ██
██       █  ██     ██  ██    ██  ██     ██  ██   ██        ██     █   ██    ██    ██     ██     ████      ████   ██   ██        ██
 ██     ██  ██     ██  ██    ██  ██     ██  ██   ██        ██     █   ██     ██   ██     ██        ██        ██  ██   ██        ██
 ██     ██  ██    ███  ██    ██  ██    ██   ██   ██        ██    ██   ██     ██   ██    ███  ██    ██  ██    ██  ██   ██        ██
  ███████    ███████    ███████  ████████   ██    ███████   ███████   ██      ██   ███████   ███████   ███████   ██    ███████  ██
   █████      █████      ██████  ██ ████    ██     █████     ████ █   ██      ██    █████      ████      ████    ██     █████   ██
                                 ██
                                 ██
                                 ██
                                 ██
*/

struct CRparams{
    double omega1;
    double omega2;
    double epsilon;
    double a;
    double c;
    double f;
};

struct CoupledRossler{
    CoupledRossler(CRparams input_params, double input_dt, double input_t_0, double input_t, double input_dump, Eigen::VectorXd input_x_0);
    ~CoupledRossler();
    Eigen::MatrixXd get_trajectory();
    Eigen::VectorXd rk4(const Eigen::VectorXd& present);
    Eigen::VectorXd coupled_rossler(const Eigen::VectorXd& present);
    Eigen::MatrixXd jacobi_matrix(const Eigen::VectorXd& state);
    double omega1;
    double omega2;
    double epsilon;
    double a;
    double c;
    double f;
    double dt;
    double t_0;
    double t;
    double dump;
    Eigen::VectorXd x_0;
    long long steps;
    long long dump_steps;
};

namespace myfunc{
    // Jacobian matrix
    Eigen::VectorXd rungeKuttaJacobian(const Eigen::VectorXd& state, const Eigen::MatrixXd& jacobian, double dt);
    Eigen::VectorXd computeDerivativeJacobian(const Eigen::VectorXd& state, const Eigen::MatrixXd& jacobian);
    Eigen::MatrixXd regularizeJacobian(const Eigen::MatrixXd& jacobian);

    // phase synchronization
    double shift(double pre_theta, double theta, double rotation_number);
    bool isSync(double a, double b, double sync_criteria, double center = 0);

    // lyapunov exponent
    template <typename Model>
    Eigen::VectorXd calcLyapunovExponent(Model M, const Eigen::MatrixXd& trajectory, int numThreads = 1){
        int numTimeSteps = trajectory.cols();
        int numVariables = trajectory.rows();
        //DataをnumThreads個に分割する(実際に分割しないが，分割したときのインデックスを計算する)
        std::vector<int> splitIndex(numThreads + 1);
        splitIndex[0] = 0;
        splitIndex[numThreads] = numTimeSteps;
        for (int i = 1; i < numThreads; ++i) {
            splitIndex[i] = numTimeSteps / numThreads * i;
        }
        //任意の直行行列を用意する
        Eigen::MatrixXd Base = Eigen::MatrixXd::Random(numVariables, numVariables);
        Eigen::HouseholderQR<Eigen::MatrixXd> qr_1(Base);
        Base = qr_1.householderQ();
        // 総和の初期化
        Eigen::VectorXd sum = Eigen::VectorXd::Zero(numVariables);
        // 次のステップ(QR分解されるもの)
        Eigen::MatrixXd next(numVariables, numVariables);
        
        #pragma omp declare reduction(vec_add : Eigen::VectorXd : omp_out += omp_in) \
            initializer(omp_priv = Eigen::VectorXd::Zero(omp_orig.size()))
        
        #pragma omp parallel for num_threads(numThreads) firstprivate(M, next, Base) shared(trajectory, splitIndex) reduction(vec_add:sum)
        for (int i = 0; i < numThreads; ++i) {
            for (int j = splitIndex[i]; j < splitIndex[i + 1]; ++j) {
                // 進捗の表示
                if (i == numThreads - 1){
                    if (j % (numTimeSteps/10000) == 0){
                        std::cout << "\r" <<  (j - splitIndex[numThreads - 1]) / static_cast<double>(splitIndex[numThreads] - splitIndex[numThreads - 1]) * 100 << "%" << std::flush;
                    }
                }
                // ヤコビアンの計算
                Eigen::MatrixXd jacobian = M.jacobian_matrix(trajectory.col(j));
                // ヤコビアンとBase(直行行列)の積を計算する
                for (int k = 0; k < numVariables; ++k) {
                    next.col(k) = myfunc::rungeKuttaJacobian(Base.col(k), jacobian, M.dt);
                }
                // QR分解を行う
                Eigen::HouseholderQR<Eigen::MatrixXd> qr(next);
                // 直交行列QでBaseを更新
                Base = qr.householderQ();
                // Rの対角成分を総和に加える
                Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
                // Rの対角成分の絶対値のlogをsumにたす
                Eigen::VectorXd diag = R.diagonal().cwiseAbs().array().log();

                sum += diag;
                // 途中経過の表示
                // if (j % 10000 == 0){
                //     std::cout << "\r" <<  sum(0) / (j+1) / M.dt << std::endl;
                // }
            }
        }
        return sum.array() / (numTimeSteps * M.dt); // 1秒あたりの変化量に変換
    }
}