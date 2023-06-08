#include <iostream>
#include <fstream>
#include <iomanip>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include "Runge_Kutta.hpp"
#include <chrono>
#include <omp.h>
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
Eigen::MatrixXd loc_max(Eigen::MatrixXd traj_abs, int obs_dim);
Eigen::VectorXcd npy2EigenVec(const char* fname);
Eigen::MatrixXd poincare_section(Eigen::MatrixXd traj_abs, int cut_dim, double cut_value);
void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname);

int main(){
    auto start = std::chrono::system_clock::now(); // timer start
    
    // generating laminar sample
    double nu = 0.00017520319481270297;
    double beta = 0.416;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 10000;
    double latter = 20;
    Eigen::VectorXcd x_0 = npy2EigenVec("../initials/beta0.416_nu0.00017520319481270297_step0.01_10000.0period_laminar.npy");
    ShellModel SM(nu, beta, f, ddt, t_0, t, latter, x_0);
    Eigen::MatrixXcd laminar = SM.get_trajectory_();
    int numRows = laminar.cols() / 10;
    Eigen::MatrixXcd laminar_sample(laminar.rows(), numRows);
    for (int i = 0; i < numRows; i++){
        int colIdx = 10 * i;
        laminar_sample.col(i) = laminar.col(colIdx);
    }
    // preserve beta of laminar
    auto beta_of_laminar = beta;

    // set up for search
    nu = 0.000173;
    beta = 0.418;
    t=50000;
    latter = 1;
    int skip = 100;
    double epsilon = 1E-1;
    int threads = omp_get_max_threads();
    std::cout << threads << "threads" << std::endl;
    int required_time = 300;


    LongLaminar LL(nu, beta, f, ddt, t_0, t, latter, x_0, laminar_sample, epsilon, skip, 100, 10, threads);
    
    Eigen::MatrixXcd trajectory = LL.get_trajectory_();
    //ラミナーの判定
    int check_times = trajectory.cols() / skip + 1;
    std::vector<int> sequence(check_times);
    for(int i =0; i < check_times; i++){
            sequence[i] = LL.isLaminarPoint_(trajectory.col(i*skip));
        }
    std::vector<int> transition_index;
    int required_consecutive_ones = static_cast<int>(required_time / ddt / skip + 0.5);
    int consecutive_ones = 0;
    for (int i = 0; i < check_times-1; i++){
        if (sequence[i] == 1){
            consecutive_ones++;
            if (consecutive_ones >= required_consecutive_ones && sequence[i+1] == 0){
            transition_index.push_back(i*skip);
            }
        }
        else{
            consecutive_ones = 0;
        }
    }
    if (transition_index.size() == 0){
        throw std::runtime_error("No transition point");
    }
    double back = 100;

    double forward = -90;
    int plot_dim1 = 3;
    int plot_dim2 = 4;
    // ラミナーが終わる前のポアンカレ写像を取得
    int loc_max_dim = 4;
    plt::figure_size(1000, 1000);
    for (const auto& index : transition_index) {
        auto poincare_section = loc_max(LL.extractor(trajectory, index, back, forward).cwiseAbs(), loc_max_dim);
        std::vector<double> x(poincare_section.cols()),y(poincare_section.cols());
        for (int i = 0; i < poincare_section.cols(); i++){
            x[i] = poincare_section(plot_dim1-1, i);
            y[i] = poincare_section(plot_dim2-1, i);
        }
        plt::scatter(x,y);
    }
    std::ostringstream oss;
    oss << "../end_laminar/beta" << beta << "_nu" << nu <<"_"<<t<<"period_from"<< back <<"to" << forward << "besides_transition-100.jpg";  // 文字列を結合する
    std::string fname = oss.str();
    plt::save(fname);

    // ラミナーが終わる近辺の軌道が欲しい時にコメントアウト外して
    // int Nth_transition = 20;
    // std::ostringstream oss;
    //  // 文字列を取得する
    // oss << "../beta" << beta << "_nu" << nu <<"_"<< std::abs(trajectory(trajectory.rows() - 1, transition_index[Nth_transition])) << ".npy";  // 文字列を結合する
    // std::string npyfname = oss.str();
    // std::cout << "Saving result to " << npyfname << std::endl;
    // EigenMt2npy(LL.extractor(trajectory, transition_index[Nth_transition], 1000, 100), npyfname);

    //timer stops
    auto end = std::chrono::system_clock::now();
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

void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname){
    Eigen::MatrixXcd transposed = Mat.transpose();
    // map to const mats in memory
    Eigen::Map<const Eigen::MatrixXcd> MOut(&transposed(0,0), transposed.cols(), transposed.rows());
    // save to np-arrays files
    cnpy::npy_save(fname, MOut.data(), {(size_t)transposed.cols(), (size_t)transposed.rows()}, "w");
}


Eigen::MatrixXd loc_max(Eigen::MatrixXd traj_abs, int loc_max_dim){
    // 条件に合えば1, 合わなければ0のベクトルを作成
    std::vector<int> binLoc_max(traj_abs.cols());
    //　最初の3点と最後の3点は条件を満たせないので0
    for (int i = 0; i < 3; ++i){
        binLoc_max[i] = 0;
        binLoc_max[binLoc_max.size()-1-i] = 0;
    }

    for (int i = 0; i < traj_abs.cols()-6; ++i){
        if (traj_abs(loc_max_dim, i+1) - traj_abs(loc_max_dim, i) > 0
        && traj_abs(loc_max_dim, i+2) - traj_abs(loc_max_dim, i+1) > 0
        && traj_abs(loc_max_dim, i+3) - traj_abs(loc_max_dim, i+2) > 0
        && traj_abs(loc_max_dim, i+4) - traj_abs(loc_max_dim, i+3) < 0
        && traj_abs(loc_max_dim, i+5) - traj_abs(loc_max_dim, i+4) < 0
        && traj_abs(loc_max_dim, i+6) - traj_abs(loc_max_dim, i+5) < 0){
            binLoc_max[i+3] = 1;
        }
    else{
        binLoc_max[i+3] = 0;
        }
    }
    Eigen::MatrixXd loc_max_point(traj_abs.rows(), binLoc_max.size());
    int col_now = 0;
    for (int i = 0; i < binLoc_max.size(); ++i){
        if (binLoc_max[i] == 1){
            loc_max_point.col(col_now) = traj_abs.col(i);
            col_now++;
        }
    }
    return loc_max_point;
}

