#include <iostream>
#include <sstream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <random>
#include <string>
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/Eigen_numpy_converter.hpp"

int main(int argc, char *argv[]) {
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    MPI_Init(&argc, &argv);
    int num_procs;
    int my_rank;
    SMparams params;
    params.nu = 4e-5;
    params.beta = 0.5;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 400;
    double dump = 0;
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../initials/beta0.5_nu4e-05_15dim.npy", true);
    ShellModel SM(params, dt, t_0, t, dump, x_0);
    int perturbed_dim = 13;
    int numThreads = omp_get_max_threads();
    double epsilon = 1e-2;
    int repetitions = 1e+1;
    int sampling_rate = 1; // sampling rate for error growth rate
    std::cout << numThreads << "threads" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> s(-1, 1);
    std::cout << std::complex<double>(s(gen), s(gen)) << std::endl;
    

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
    Eigen::VectorXd time(SM.steps + 1); //　時間を格納するベクトル
    int counter = 0; // just for progress bar
    #pragma omp declare reduction(+ : Eigen::VectorXd : omp_out = omp_out + omp_in) \
        initializer(omp_priv = Eigen::VectorXd::Zero(omp_orig.size()))

    #pragma omp parallel for num_threads(numThreads) schedule(dynamic) firstprivate(SM, perturbed_dim, repetitions, epsilon) shared(time, counter) reduction(+ : average_errors)
    for(int i = 0; i < repetitions; i++){
        SM.x_0 = myfunc::multi_scale_perturbation(SM.x_0, -3, -2); // 初期値をランダムに与える
        // ある程度まともな値になるように初期値を更新
        int sec_500 = static_cast<int>(500 / SM.dt);
        for (int j = 0; j < sec_500; j++) {
            SM.x_0 = SM.rk4(SM.x_0);
        }

        // ここから本番
        //まずは元の軌道を計算
        Eigen::MatrixXcd original = SM.get_trajectory();
        if (i == 0) {
            time = original.bottomRows(1).cwiseAbs().row(0);
        }

        // 初期値の指定した変数にだけ摂動を与える
        SM.x_0(perturbed_dim - 1) += SM.x_0(perturbed_dim - 1) * std::complex<double>(s(gen), s(gen)) * epsilon;
        // SM.x_0(perturbed_dim - 1) += epsilon;
        Eigen::MatrixXcd another = SM.get_trajectory();

        // 誤差の計算
        //差をとる
        Eigen::MatrixXcd diff = original.topRows(original.rows() - 1) - another.topRows(another.rows() - 1);
        Eigen::VectorXd errors(SM.steps+1);
        // 各列のノルムを計算
        for (int j = 0; j < SM.steps; j++) {
            errors(j) = diff.col(j).norm();
        }

        average_errors += errors / repetitions;

        #pragma atomic
        counter++; // just for progress bar
        if (omp_get_thread_num() == 0){
            std::cout << "\r processing..." << counter << "/" << repetitions << std::flush;
        }
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
    oss << "../../error_growth/error-rate_beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions << "sampling" << sampling_rate << ".png";  // 文字列を結合する
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
    oss << "../../error_growth/time-error_beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions << "sampling" << sampling_rate << ".png";  // 文字列を結合する
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
    oss << "../../error_growth/time-rate_beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions << "sampling" << sampling_rate << ".png";  // 文字列を結合する
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
    oss2 << "../../error_growth/beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions << "sampling" << sampling_rate << ".npy";  // 文字列を結合する
    std::string fname = oss2.str(); // 文字列を取得する
    std::cout << "Saving result to " << fname << std::endl;
    EigenMat2npy(result, fname);
    if (my_rank == 0) {
        myfunc::duration(start);
    }
}