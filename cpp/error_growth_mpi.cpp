/**
 * @file error_growth_mpi.cpp
 * @author Hibiki Kato
 * @brief error growthの計算をMPIで並列化
 * @version 0.1
 * @date 2024-01-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <iostream>
#include <sstream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <mpi.h>
#include <random>
#include <string>
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

int main(int argc, char *argv[]) {
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    
    MPI_Init(&argc, &argv);
    int num_procs;
    int my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    SMparams params;
    params.nu = 4e-5;
    params.beta = 0.5;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 400;
    double dump = 0;
    Eigen::VectorXcd x_0 = Eigen::VectorXcd::Random(15) * 1e-3;
    ShellModel SM(params, dt, t_0, t, dump, x_0);
    int perturbed_dim = 13;
    int numThreads = omp_get_max_threads();
    double epsilon = 1e-2;
    int repetitions = 1e+3;
    int sampling_rate = 1; // sampling rate for error growth rate
    if (my_rank == 0) std::cout << numThreads << "threads" << std::endl;
    //MPI用にrepetitionsを分割
    std::vector<int> ite(num_procs);
    for (int i = 0; i < num_procs - 1; i++) {
        ite[i] = static_cast<int>(repetitions / num_procs);
    }
    ite[num_procs - 1] = repetitions - (num_procs - 1) * static_cast<int>(repetitions / num_procs);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> s(-1, 1);

    std::cout << "my_rank: " << my_rank << " num_procs: " << num_procs << " loop"<< ite[my_rank] << std::endl;

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
    int counter = 0;
    #pragma omp declare reduction(+ : Eigen::VectorXd : omp_out = omp_out + omp_in) \
        initializer(omp_priv = Eigen::VectorXd::Zero(omp_orig.size()))

    #pragma omp parallel for num_threads(numThreads) schedule(dynamic) firstprivate(SM, perturbed_dim, epsilon, ite, my_rank) shared(time, counter) reduction(+ : average_errors)
    for(int i = 0; i < ite[my_rank]; i++){
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
        for (int j = 0; j < SM.steps+1; j++) {
            errors(j) = diff.col(j).norm();
        }

        average_errors += errors / repetitions;
        #pragma omp atomic
        counter++;

        if (my_rank == 0 && omp_get_thread_num() == 0){
            std::cout << "count " << counter << std::endl;
        }
    }
    MPI_Reduce(
        &average_errors(0), // sendbuf
        &average_errors(0), // recvbuf
        average_errors.size(), // count
        MPI_DOUBLE, // datatype
        MPI_SUM, // op
        0, // root
        MPI_COMM_WORLD // comm
    );
    //以下並列化しないのでmy_rank=0のプロセスだけ残す
    if (my_rank!= 0){
        MPI_Finalize();
        return 0;
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
    // error_vecとerror_rate_vecとtime_vecを一時ファイルにまとめて保存
    std::string data_file = "tmp.dat";
    std::ofstream ofs(data_file);
    for (int i = 0; i < error_vec.size(); i++) {
        ofs << error_vec[i] << "\t" << growth_rate_vec[i] << "\t" << time_vec[i] << std::endl;
    }
    ofs.close();

    // gnuplotでプロット
    std::ostringstream oss;

    // エラー : エラー成長率
    oss << "../../error_growth/error-rate_beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions << "sampling" << sampling_rate << ".png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    std::ostringstream gnuplotCmd;
    gnuplotCmd << "gnuplot -persist -e \"";
    gnuplotCmd << "set term png size 1200,800 font 'Times New Roman,20'; ";
    gnuplotCmd << "set output '" << plotfname << "'; ";
    gnuplotCmd << "set xlabel 'E'; ";
    gnuplotCmd << "set ylabel 'E/E'; ";
    gnuplotCmd << "set logscale x; set format x '10^{%L}';";
    gnuplotCmd << "set autoscale; ";
    gnuplotCmd << "unset key; ";
    gnuplotCmd << "plot '"<< data_file << "' u 1:2 with points pt 7 lc 'blue'; ";
    gnuplotCmd << "replot; ";
    gnuplotCmd << "set output; ";
    gnuplotCmd << "set term qt; ";
    gnuplotCmd << "exit;\"";
    // std::cout << gnuplotCmd.str() << std::endl;
    system(gnuplotCmd.str().c_str());

    // 時間 : エラー
    oss.str("");
    oss << "../../error_growth/time-error_beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions << "sampling" << sampling_rate << ".png";  // 文字列を結合する
    std::string plotfname1 = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname1 << std::endl;
    gnuplotCmd.str("");
    gnuplotCmd << "gnuplot -persist -e \"";
    gnuplotCmd << "set term png size 1200,800 font 'Times New Roman,20'; ";
    gnuplotCmd << "set output '" << plotfname1 << "'; ";
    gnuplotCmd << "set xlabel 't'; ";
    gnuplotCmd << "set ylabel 'E'; ";
    gnuplotCmd << "set logscale y;  set format y '10^{%L}';";
    gnuplotCmd << "unset key; ";
    gnuplotCmd << "plot '"<< data_file << "' u 3:1 with points pt 7 lc 'blue'; ";
    gnuplotCmd << "replot; ";
    gnuplotCmd << "set output; ";
    gnuplotCmd << "set term qt; ";
    gnuplotCmd << "exit;\"";
    // std::cout << gnuplotCmd.str() << std::endl;
    system(gnuplotCmd.str().c_str());

    // 時間 : エラー成長率
    oss.str("");
    oss << "../../error_growth/time-rate_beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions << "sampling" << sampling_rate << ".png";  // 文字列を結合する
    std::string plotfname2 = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname2 << std::endl;
    gnuplotCmd.str("");
    gnuplotCmd << "gnuplot -persist -e \"";
    gnuplotCmd << "set term png size 1200,800 font 'Times New Roman,20'; ";
    gnuplotCmd << "set output '" << plotfname2 << "'; ";
    gnuplotCmd << "set xlabel 't'; ";
    gnuplotCmd << "set ylabel 'E/E'; ";
    gnuplotCmd << "unset key; ";
    gnuplotCmd << "plot '"<< data_file << "' u 3:2 with points pt 7 lc 'blue'; ";
    gnuplotCmd << "replot; ";
    gnuplotCmd << "set output; ";
    gnuplotCmd << "set term qt; ";
    gnuplotCmd << "exit;\"";
    // std::cout << gnuplotCmd.str() << std::endl;
    system(gnuplotCmd.str().c_str());
    
    /*
     ████                          ██            ██        ██
    ██  ██                         ██            ██        ██
    █    █   ███  ██   █  ███     ████  ███     ████ █  ██████
    ██      █  ██  █  ██ ██  █     ██  ██  █     ██  ██ █  ██
     ████       █  █  █  █   █     ██  █   ██    ██   ███  ██
        ██   ████  ██ █  █████     ██  █    █    ██   ██   ██
    █    █  █   █   ███  █         ██  █   ██    ██   ███  ██
    ██  ██  █  ██   ██   ██  █     ██  ██  █     ██  ██ █  ██
     ████   █████   ██    ████      ██  ███       ██ █  ██  ██
    */
    std::ostringstream oss2;
    oss2 << "../../error_growth/beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions << "sampling" << sampling_rate << ".dat";  // 文字列を結合する
    std::string fname = oss2.str(); // 文字列を取得する
    std::cout << "Saving result to " << fname << std::endl;
    myfunc::duration(start);
    std::ofstream ofs2(fname);
    if (!ofs2) {
        std::cout << "Error: cannot open " << fname << std::endl;
        return 1;
    }
    for (int i = 0; i < error_vec.size(); i++) {
        ofs2 << error_vec[i] << "\t" << growth_rate_vec[i] << "\t" << time_vec[i] << std::endl;
    }
    ofs2.close();
    MPI_Finalize();
    return 0;
}