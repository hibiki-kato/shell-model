#include <iostream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <omp.h>
#include <chrono>
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 4e-5;
    params.beta = 0.5;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 400;
    double t_initial = 26;
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../initials/beta0.5_nu4e-05_15dim.npy", true);
    int dim = x_0.size();
    ShellModel SM(params, dt, t_0, t, t_initial, x_0);
    int perturbed_dim = 13;
    int numThreads = omp_get_max_threads();
    double epsilon = 1e-5;
    int repetitions = 5e+6;
    int sampling_rate = 1; // sampling rate for error growth rate
    std::cout << numThreads << "threads" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> s(-1, 1);
    

    /*
                   █
     ████          █              ██
    ██   █         █              ██
    █    ██  ███   █   ███        █ █  ██   █  ███   ███  ███    ████   ███
    █       █  ██  █  ██  █      ██ █   █  ██ ██  █  ██  █  ██  ██  █  ██  █
    █           █  █  █          █  ██  █  █  █   █  █       █  █   █  █   █
    █        ████  █  █         ██   █  ██ █  █████  █    ████  █   █  █████
    █    ██ █   █  █  █         ██████   ███  █      █   █   █  █   █  █
    ██   █  █  ██  █  ██  █     █    ██  ██   ██  █  █   █  ██  ██  █  ██  █
     ████   █████  █   ███     ██     █  ██    ████  █   █████   ████   ████
                                                                    █
                                                                █   █
                                                                 ███
    */
    Eigen::VectorXd average_errors(SM.steps + 1); // 誤差の平均を格納するベクトル
    int counter = 0; // just for progress bar
    int invalid_counter = 0; // for counting invalid (inf or nan) errors
    #pragma omp declare reduction(+ : Eigen::VectorXd : omp_out = omp_out + omp_in) \
        initializer(omp_priv = Eigen::VectorXd::Zero(omp_orig.size()))

    #pragma omp parallel for num_threads(numThreads) schedule(dynamic) firstprivate(SM, perturbed_dim, repetitions, epsilon) shared(counter, invalid_counter) reduction(+ : average_errors)
    for(int i = 0; i < repetitions; i++){
        SM.x_0 = myfunc::multi_scale_perturbation(SM.x_0, -3, -2); // 初期値をランダムに与える
        // ある程度まともな値になるように初期値を更新
        for (int j = 0; j < 5e+4; j++) {
            SM.x_0 = SM.rk4(SM.x_0);
        }

        // ここから本番
        // original
        Eigen::MatrixXcd original(dim, SM.steps + 1); // 誤差を計算するための元の軌道
        // それぞれ初期値を代入
        original.col(0) = SM.x_0;
        for (int j = 0; j < SM.dump_steps; j++) {
            original.col(j + 1) = SM.rk4(original.col(j));
        }
        for (int j = SM.dump_steps; j < SM.steps; j++) {
            // ヤコビアンを作用させる用の実ベクトル
            Eigen::VectorXd state = Eigen::VectorXd::Zero(dim*2);
            for (int k = 0; k < dim; ++k) {
                state(2*k) = original(k, j).real();
                state(2*k+1) = original(k, j).imag();
            }
            Eigen::MatrixXd jacobian_matrix = SM.jacobian_matrix(state);
            // ヤコビ行列による時間発展
            state = myfunc::rungeKuttaJacobian(state, jacobian_matrix, SM.dt);
            // 複素ベクトルに戻す
            for (int k = 0; k < dim; k++){
                std::complex<double> tmp(state(2*k), state(2*k+1));
                original(k, j+1) = tmp;
            }
        }
        // another
        Eigen::MatrixXcd another(dim, SM.steps + 1); // 誤差を計算するための別の軌道
        SM.x_0(perturbed_dim - 1) += SM.x_0(perturbed_dim - 1) * std::complex<double>(s(gen), s(gen)) * epsilon;
        // SM.x_0(perturbed_dim - 1) += epsilon;
        another.col(0) = SM.x_0;
        for (int j = 0; j < SM.dump_steps; j++) {
            another.col(j + 1) = SM.rk4(another.col(j));
        }
        for (int j = SM.dump_steps; j < SM.steps; j++) {
            // ヤコビアンを作用させる用の実ベクトル
            Eigen::VectorXd state = Eigen::VectorXd::Zero(dim*2);
            for (int k = 0; k < dim; ++k) {
                state(2*k) = another(k, j).real();
                state(2*k+1) = another(k, j).imag();
            }
            Eigen::MatrixXd jacobian_matrix = SM.jacobian_matrix(state);
            // ヤコビ行列による時間発展
            state = myfunc::rungeKuttaJacobian(state, jacobian_matrix, SM.dt);
            // 複素ベクトルに戻す
            for (int k = 0; k < dim; k++){
                std::complex<double> tmp(state(2*k), state(2*k+1));
                another(k, j+1) = tmp;
            }
        }


        // 誤差の計算
        //差をとる
        Eigen::MatrixXcd diff = original - another;
        Eigen::VectorXd errors(SM.steps+1);
        // 各列のノルムを計算
        for (int j = 0; j < SM.steps; j++) {
            errors(j) = diff.col(j).norm();
        }
        //errorsにinfがあるかどうか
        if (errors.hasNaN()){
            #pragma omp atomic
            invalid_counter++;
        }else{
            average_errors += errors;
        }

        #pragma atomic
        counter++; // just for progress bar
        if (omp_get_thread_num() == 0){
            std::cout << "\r processing..." << counter << "/" << repetitions << std::flush;
        }
    }
    std::cout << invalid_counter <<"/" << repetitions << " diverged" << std::endl;
    average_errors /= (repetitions - invalid_counter);
    Eigen::VectorXd time(SM.steps + 1); //　時間を格納するベクトル
    time(0) = SM.t_0;
    for (int i = 0; i < SM.steps; i++) {
        time(i + 1) = time(i) + SM.dt;
    }

    /*
                                     █                                                █                                            █
     ████                            █  █                    ███        ████          █                                        ██  █                    ██
    ██  ██                           █                      ██  █      ██   █         █                                        ██  █                    ██
    █    █   ███   █████ ███  █████  █  █  █ ███   ████     ██  █      █    ██  ███   █   ███       ████  ███  ███  ██  ██  █ ████ █████     ███  ███  ████  ███
    ██      █  ██  ██  ██  █  ██  █  █  █  ██  █  ██  █      ███       █       █  ██  █  ██  █     ██  █  ██  ██  █  █  ██  █  ██  ██  █     ██  █  ██  ██  ██  █
     ████       █  █   █   ██ █   ██ █  █  █   █  █   █      ██        █           █  █  █         █   █  █   █   ██ █  ██  █  ██  █   █     █       █  ██  █   █
        ██   ████  █   █   ██ █   ██ █  █  █   █  █   █     █  █ █     █        ████  █  █         █   █  █   █    █ █ █ █ ██  ██  █   █     █    ████  ██  █████
    █    █  █   █  █   █   ██ █   ██ █  █  █   █  █   █     █  ███     █    ██ █   █  █  █         █   █  █   █   ██ ███  ██   ██  █   █     █   █   █  ██  █
    ██  ██  █  ██  █   █   ██ ██  █  █  █  █   █  ██  █     █   ██     ██   █  █  ██  █  ██  █     ██  █  █   ██  █   ██  ██   ██  █   █     █   █  ██  ██  ██  █
     ████   █████  █   █   ██ █████  █  █  █   █   ████      ██████     ████   █████  █   ███       ████  █    ███    █   ██    ██ █   █     █   █████   ██  ████
                              █                       █                                                █
                              █                   █   █                                            █   █
                              █                    ███                                              ███
    */

    // sampling
    int sampling_num = average_errors.size() / sampling_rate;
    Eigen::VectorXd sampled_time(sampling_num);
    Eigen::VectorXd sampled_errors(sampling_num);
    for (int i = 0; i < sampling_num; i++) {
        sampled_time(i) = time(sampling_rate * i);
        sampled_errors(i) = average_errors(sampling_rate * i);
    }

    // error growth rate
    // Eigen::VectorXd growth_rate = (sampled_errors.tail(sampled_errors.size() - 2).array().log() - sampled_errors.head(sampled_errors.size() - 2).array().log()) / (SM.dt*sampling_rate*2);
    Eigen::VectorXd growth_rate(sampled_errors.size() - 2);
    for (int i = 0; i < sampled_errors.size() - 2; i++) {
        growth_rate(i) = (std::log(sampled_errors(i + 2)) - std::log(sampled_errors(i))) / (SM.dt*sampling_rate*2);
    }

    /*
            █
    ██████  █         ██  ██  █
    █    █  █         ██  ██
    █    ██ █   ███  ████████ █  █ ███   ████
    █    ██ █  ██  █  ██  ██  █  ██  █  ██  █
    ██████  █  █   ██ ██  ██  █  █   █  █   █
    █       █  █    █ ██  ██  █  █   █  █   █
    █       █  █   ██ ██  ██  █  █   █  █   █
    █       █  ██  █  ██  ██  █  █   █  ██  █
    █       █   ███    ██  ██ █  █   █   ████
                                            █
                                        █   █
                                         ███
    */

    std::vector<double> error_vec(sampled_errors.data(), sampled_errors.data() + sampled_errors.size());
    std::vector<double> growth_rate_vec(growth_rate.data(), growth_rate.data() + growth_rate.size());
    std::vector<double> time_vec(sampled_time.data(), sampled_time.data() + sampled_time.size());
    // // error_vecとtime_vecの先頭と末尾を1秒削除
    int start_ = static_cast<int>(1.0 / (SM.dt * sampling_rate));
    // for (int i = 0; i < start_; i++) {
    //     error_vec.erase(error_vec.begin());
    //     growth_rate_vec.erase(growth_rate_vec.begin());
    //     time_vec.erase(time_vec.begin());
    // }
    error_vec.erase(error_vec.begin());
    error_vec.pop_back();
    time_vec.erase(time_vec.begin());
    time_vec.pop_back();
    //check the size
    // std::cout << error_vec.size() << "   "  << growth_rate_vec.size() << " " << time_vec.size()<<  std::endl;

    std::ostringstream oss;
    // エラー : エラー成長率
    plt::figure_size(1200, 800);
    plt::xscale("log");
    plt::xlabel("E");
    plt::ylabel("E'/E");
    plt::scatter(error_vec, growth_rate_vec);
    oss << "../../error_growth/linear_error-rate_beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions - invalid_counter << "sampling" << sampling_rate << "perturbed_dim" << perturbed_dim << ".png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);
    plt::close();

    // エラー : 時間
    plt::figure_size(1200, 800);
    plt::yscale("log");
    plt::scatter(time_vec, error_vec);
    plt::xlabel("t");
    plt::ylabel("E");
    oss.str("");
    oss << "../../error_growth/linear_time-error_beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions - invalid_counter << "sampling" << sampling_rate << "perturbed_dim" << perturbed_dim << ".png";  // 文字列を結合する
    std::string plotfname1 = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname1 << std::endl;
    plt::save(plotfname1);
    plt::close();

    // エラー成長率 : 時間
    plt::figure_size(1200, 800);
    // plt::xscale("log");
    plt::scatter(time_vec, growth_rate_vec);
    plt::xlabel("t");
    plt::ylabel("E'/E");
    oss.str("");
    oss << "../../error_growth/linear_time-rate_beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions - invalid_counter << "sampling" << sampling_rate << "perturbed_dim" << perturbed_dim << ".png";  // 文字列を結合する
    std::string plotfname2 = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname2 << std::endl;
    plt::save(plotfname2);

    /*
     ████                          ██
    ██  ██                         ██
    █    █   ███  ██   █  ███     ████  ███      █ ███  █████ ██   █
    ██      █  ██  █  ██ ██  █     ██  ██  █     ██  █  ██  █  █  ██
     ████       █  █  █  █   █     ██  █   ██    █   █  █   ██ █  █
        ██   ████  ██ █  █████     ██  █    █    █   █  █   ██ ██ █
    █    █  █   █   ███  █         ██  █   ██    █   █  █   ██  ███
    ██  ██  █  ██   ██   ██  █     ██  ██  █     █   █  ██  █   ██
     ████   █████   ██    ████      ██  ███      █   █  █████   ██
                                                        █       ██
                                                        █       █
                                                        █     ██
    */
    Eigen::MatrixXd result(3, error_vec.size());
    for (int i = 0; i < error_vec.size(); i++) {
        result(0, i) = error_vec[i];
        result(1, i) = growth_rate_vec[i];
        result(2, i) = time_vec[i];
    }
    std::ostringstream oss2;
    oss2 << "../../error_growth/linear_beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions - invalid_counter << "sampling" << sampling_rate << "perturbed_dim" << perturbed_dim << ".npy";  // 文字列を結合する
    std::string fname = oss2.str(); // 文字列を取得する
    std::cout << "Saving result to " << fname << std::endl;
    EigenMat2npy(result, fname);
    myfunc::duration(start);
}