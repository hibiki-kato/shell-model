#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include "Runge_Kutta.hpp"
#include <chrono>
#include <random>
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
void EigenVec2npy(Eigen::VectorXd Vec, std::string fname);
Eigen::VectorXcd npy2EigenVec(const char* fname);
Eigen::VectorXcd perturbation(Eigen::VectorXcd state,  std::vector<int> dim, int s_min = -1, int s_max = -1);


int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    
    double nu = 1e-5;
    double beta = 0.5;
    std::complex<double> f = std::complex<double>(1.0,0.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 400;
    double latter = 1;
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.5_nu1e-05_15dim_period.npy");
    ShellModel SM(nu, beta, f, ddt, t_0, t, latter, x_0);
    std::vector<int> perturbed_dim = {13};
    int threads = omp_get_max_threads();
    int repetitions = 1000;
    int sampling = 100; // sampling rate of 
    std::cout << threads << "threads" << std::endl;
    std::ostringstream oss;
    
    int counter = 0; // just for progress bar
    Eigen::VectorXd average_errors(SM.get_steps_()+1);
    Eigen::VectorXd time(SM.get_steps_()-6);
    #pragma omp parallel for num_threads(threads) schedule(dynamic) shared(average_errors, time) firstprivate(SM, counter)
    for(int i = 0; i < repetitions; i++){
        if(omp_get_thread_num() == 0)
        {
        std::cout << "\r processing..." << counter * threads << "/" << repetitions << std::flush;
        counter++;
        }
        ShellModel SM_origin = SM;
        //1からx_0.size()のベクトルの作成
        std::vector<int> range(x_0.size());
        std::iota(range.begin(), range.end(), 1); // iota: 連番を作成する
        ShellModel SM_another = SM;
        Eigen::VectorXcd perturbed_x_0 = SM_another.get_x_0_();
        perturbed_x_0(12) += 1e-5;
         // create perturbed init value
        SM_another.set_x_0_(perturbed_x_0); // set above
        
        Eigen::MatrixXcd origin = SM_origin.get_trajectory_();
        Eigen::MatrixXcd another = SM_another.get_trajectory_();
        Eigen::VectorXd errors = (origin.topRows(origin.rows() - 1) - another.topRows(another.rows() - 1)).cwiseAbs2().colwise().sum().cwiseSqrt();
        #pragma omp critical
        {
            average_errors += errors / repetitions;
            SM.set_x_0_(another.block(0, another.cols()-1, another.rows()-1, 1)); // set another's last state as next initial value
        }
        if (i==0){
            time = origin.row(15).cwiseAbs().segment(1, time.size()+1);
        }
    }

    // EigenVec2npy(average_errors, "../../test.npy");

    Eigen::VectorXd average_error_growth_rates = (average_errors.tail(average_errors.size() - 2).array().log() - average_errors.head(average_errors.size() - 2).array().log()) / (SM.get_ddt_()*2);
    // Eigen::VectorXd average_error_growth_rates = (average_errors.tail(average_errors.size() - 6).array().log() - 9*average_errors.segment(5, average_errors.size() - 6).array().log() + 45*average_errors.segment(4, average_errors.size() - 6).array().log() - 45*average_errors.segment(2, average_errors.size() - 6).array().log() + 9*average_errors.segment(1, average_errors.size() - 6).array().log() - average_errors.head(average_errors.size() - 6)).array().log() / (60*ddt);
    Eigen::VectorXd average_errors_resized = average_errors.segment(1, average_errors.size() - 6);
    std::cout << average_error_growth_rates << std::endl;
    std::vector<double> errors_(average_errors_resized.data(), average_errors_resized.data() + average_errors_resized.size());
    std::vector<double> error_growth_rates_(average_error_growth_rates.data(), average_error_growth_rates.data() + average_error_growth_rates.size());
    std::vector<double> time_(time.data(), time.data()+time.size());

    std::cout << errors_.size() << "   "  << error_growth_rates_.size() << " " << time_.size()<<  std::endl;

    // エラー : エラー成長率
    plt::figure_size(1000, 1000);
    plt::xscale("log");
    plt::xlabel("E");
    plt::ylabel("E'/E");
    plt::scatter(errors_, error_growth_rates_);
    oss << "../../error_growth/beta_" << beta << "nu_" << nu << "error"<< t / latter <<"period" << repetitions << "repeat.png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);
    plt::close();

    // エラー : 時間
    plt::figure_size(1000, 1000);
    // plt::xscale("log");
    plt::scatter(time_, errors_);
    plt::xlabel("t");
    plt::ylabel("E");
    oss.str("");
    oss << "../../error_growth/error_beta_" << beta << "nu_" << nu << "error"<< t / latter <<"period" << repetitions << "repeat.png";  // 文字列を結合する
    std::string plotfname1 = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname1 << std::endl;
    plt::save(plotfname1);
    plt::close();

    // エラー成長率 : 時間
    plt::figure_size(1000, 1000);
    // plt::xscale("log");
    plt::scatter(time_, error_growth_rates_);
    plt::xlabel("t");
    plt::ylabel("E'/E");
    oss.str("");
    oss << "../../error_growth/error_rate_beta_" << beta << "nu_" << nu << "error"<< t / latter <<"period" << repetitions << "repeat.png";  // 文字列を結合する
    std::string plotfname2 = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname2 << std::endl;
    plt::save(plotfname2);

    auto end = std::chrono::system_clock::now();  // 計測終了時間
    int hours = std::chrono::duration_cast<std::chrono::hours>(end-start).count(); //処理に要した時間を変換
    int minutes = std::chrono::duration_cast<std::chrono::minutes>(end-start).count(); //処理に要した時間を変換
    int seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count(); //処理に要した時間を変換
    int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間を変換
    std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;
}

Eigen::VectorXcd npy2EigenVec(const char* fname){
    std::string fname_str(fname);
    cnpy::NpyArray arr = cnpy::npy_load(fname_str);
    if (arr.word_size != sizeof(std::complex<double>)) {
        throw std::runtime_error("Unsupported data type in the npy file.");
    }
    std::complex<double>* data = arr.data<std::complex<double>>();
    Eigen::Map<Eigen::VectorXcd> vec(data, arr.shape[0]);
    return vec;
}

void EigenMt2npy(Eigen::MatrixXd Mat, std::string fname){
    Eigen::MatrixXd transposed = Mat.transpose();
    // map to const mats in memory
    Eigen::Map<const Eigen::MatrixXd> MOut(&transposed(0,0), transposed.cols(), transposed.rows());
    // save to npy file
    cnpy::npy_save(fname, MOut.data(), {(size_t)transposed.cols(), (size_t)transposed.rows()}, "w");
}

Eigen::VectorXcd perturbation(Eigen::VectorXcd state, std::vector<int> dim, int s_min, int s_max){
    Eigen::VectorXcd perturbed = state;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> s(-1, 1);
    std::uniform_real_distribution<double> dis(s_min, s_max);

    Eigen::VectorXd unit = Eigen::VectorXd::Ones(state.rows());
    for(int shell : dim){
        perturbed(shell-1) += state(shell-1) * s(gen) * std::pow(10, dis(gen)); //元の値 * (-1, 1)の一様分布 * 10^(指定の範囲から一様分布に従い選ぶ)　を雪道として与える
    }

    return perturbed;
}

void EigenVec2npy(Eigen::VectorXd Vec, std::string fname){
    std::vector<double> x(Vec.size());
    for(int i=0;i<Vec.size();i++){
        x[i]=Vec(i);
    }
    cnpy::npy_save(fname, &x[0], {(size_t)Vec.size()}, "w");
}