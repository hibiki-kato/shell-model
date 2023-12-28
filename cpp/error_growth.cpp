#include <iostream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <chrono>
#include "cnpy/cnpy.h"
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
    params.f = std::complex<double>(1.0,0.0) * 5.0 * 0.001;
    double dt = 0.001;
    double t_0 = 0;
    double t = 400;
    double dump = 0;
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../initials/beta0.5_nu1e-05_15dim_period.npy", true);
    ShellModel SM(params, dt, t_0, t, dump, x_0);
    int perturbed_dim = 13;
    int numThreads = omp_get_max_threads();
    double epsilon = 1e-5;
    int repetitions = 1000;
    int sampling_rate = 1000; // sampling rate for error growth rate
    std::cout << numThreads << "threads" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> s(-1, 1);
    Eigen::VectorXd time(SM.steps + 1); //　時間を格納するベクトル
    Eigen::MatrixXd errors(x_0.size(), SM.steps + 1);// 各試行の誤差を格納する行列
    int counter = 0; // just for progress bar

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

    #pragma omp parallel for num_threads(numThreads) schedule(dynamic) firstprivate(SM, perturbed_dim, repetitions) shared(errors, time, counter)
    for(int i = 0; i < repetitions; i++){
        SM.x_0 = myfunc::multi_scale_perturbation(SM.x_0, -1, 0); // 初期値をランダムに与える
        // ある程度まともな値になるように初期値を更新
        for (int j = 0; j < 1e+5; j++) {
            SM.x_0 = SM.rk4(SM.x_0);
        }

        // ここから本番
        //まずは元の軌道を計算
        Eigen::MatrixXcd origin = SM.get_trajectory();

        // 初期値の指定した変数にだけ摂動を与える
        SM.x_0(perturbed_dim - 1) += epsilon * std::complex<double>(s(gen), s(gen));
        Eigen::MatrixXcd another = SM.get_trajectory();

        #pragma atomic
        counter++; // just for progress bar
        #pragma omp critical
        {
            errors += (origin.topRows(origin.rows() - 1) - another.topRows(another.rows() - 1)).cwiseAbs2() / repetitions;
            std::cout << "\r processing..." << counter << "/" << repetitions << std::flush;
        }
        if (i == 0) {
            time = origin.bottomRows(1).cwiseAbs().row(0);
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

    Eigen::VectorXd average_errors = errors.colwise().sum().array().sqrt();

    // sampling
    int sampling_num = average_errors.size() / sampling_rate;
    Eigen::VectorXd sampled_time(sampling_num);
    Eigen::VectorXd sampled_errors(sampling_num);
    for (int i = 0; i < sampling_num; i++) {
        sampled_time(i) = time(sampling_rate * i);
        sampled_errors(i) = average_errors(sampling_rate * i);
    }

    // error growth rate
    Eigen::VectorXd growth_rate = (sampled_errors.tail(sampled_errors.size() - 2).array().log() - sampled_errors.head(sampled_errors.size() - 2).array().log()) / (SM.dt*sampling_rate*2);
    // Eigen::VectorXd growth_rate = (sampled_errors.tail(sampled_errors.size() - 6).array().log() - 9*sampled_errors.segment(5, sampled_errors.size() - 6).array().log() + 45*sampled_errors.segment(4, sampled_errors.size() - 6).array().log() - 45*sampled_errors.segment(2, sampled_errors.size() - 6).array().log() + 9*sampled_errors.segment(1, sampled_errors.size() - 6).array().log() - sampled_errors.head(sampled_errors.size() - 6)).array().log() / (60*ddt);
    std::cout << "here" << std::endl;
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
    // error_vecとtime_vecの先頭と末尾を削除
    error_vec.erase(error_vec.begin());
    error_vec.pop_back();
    time_vec.erase(time_vec.begin());
    time_vec.pop_back();
    //check the size
    // std::cout << error_vec.size() << "   "  << growth_rate_vec.size() << " " << time_vec.size()<<  std::endl;

    std::ostringstream oss;
    // エラー : エラー成長率
    plt::figure_size(1000, 1000);
    plt::xscale("log");
    plt::xlabel("E");
    plt::ylabel("E'/E");
    plt::scatter(error_vec, growth_rate_vec);
    oss << "../../error_growth/error-rate_beta" << params.beta << "nu" << params.nu << "t" << t << "repeat" << repetitions << "sampling" << sampling_rate << ".png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);
    plt::close();

    // エラー : 時間
    plt::figure_size(1000, 1000);
    // plt::xscale("log");
    plt::scatter(time_vec, error_vec);
    plt::xlabel("t");
    plt::ylabel("E");
    oss.str("");
    oss << "../../error_growth/time-error_beta" << params.beta << "nu" << params.nu << "t" << t << "repeat" << repetitions << "sampling" << sampling_rate << ".png";  // 文字列を結合する
    std::string plotfname1 = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname1 << std::endl;
    plt::save(plotfname1);
    plt::close();

    // エラー成長率 : 時間
    plt::figure_size(1000, 1000);
    // plt::xscale("log");
    plt::scatter(time_vec, growth_rate_vec);
    plt::xlabel("t");
    plt::ylabel("E'/E");
    oss.str("");
    oss << "../../error_growth/time-rate_beta" << params.beta << "nu" << params.nu << "t" << t << "repeat" << repetitions << "sampling" << sampling_rate << ".png";  // 文字列を結合する
    std::string plotfname2 = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname2 << std::endl;
    plt::save(plotfname2);

    myfunc::duration(start);
}