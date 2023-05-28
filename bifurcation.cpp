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
Eigen::MatrixXd loc_max(Eigen::MatrixXd Mt, int obs_dim);
Eigen::VectorXcd npy2EigenVec(const char* fname);


int main(){
    auto start = std::chrono::system_clock::now(); // timer start
    double nu = 0.00017;
    double beta = 0.425;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 10000;
    double latter = 10;
    Eigen::VectorXcd x_0 = npy2EigenVec("../initials/beta0.416_nu0.00017520319481270297_step0.01_10000.0period_laminar.npy");
    int threads = omp_get_max_threads();
    std::cout << threads << "threads" << std::endl;

    int param_steps = 100;
    double beta_begin = 0.414;
    double beta_end = 0.441;
    double nu_begin = 0.00018;
    double nu_end = 0.000135;
    int loc_max_dim = 3;
    int target_dim = 4;

    ShellModel SM(nu, beta, f, ddt, t_0, t, latter, x_0);
    Eigen::VectorXd betas = Eigen::VectorXd::LinSpaced(param_steps, beta_begin, beta_end);
    Eigen::VectorXd nus = Eigen::VectorXd::LinSpaced(param_steps, nu_begin, nu_end);

    std::ostringstream oss;
    oss << "../bif_data/bif_" << beta_begin <<"to"<< beta_end << "_nu" << nu_begin <<"to" << nu_end << "_" << param_steps << "steps_period" << t-t_0 << "_latter_"<< std::setprecision(2) << 1 / latter << ".txt";  // 文字列を結合する
    std::string fname = oss.str();
    std::ofstream file(fname);
    if (!file) {
        std::cerr << "ファイルを開けませんでした。" << std::endl;
        return 1;
    }

    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < param_steps; i++) {
        if (omp_get_thread_num() ==0){
            std::cout << "\r 現在" << i * threads << "/" << param_steps << std::flush;
        }
        ShellModel local_SM = SM;
        local_SM.set_beta_(betas(i));
        local_SM.set_nu_(nus(i));
        Eigen::MatrixXcd trajectory = local_SM.get_trajectory_();
        auto poincare_section = loc_max(trajectory.cwiseAbs(), loc_max_dim);
        
        #pragma omp critical
        {
            file << betas(i) << " " << nus(i) << " ";
            
            //target_dim - 1行目を出力
            for (int j = 0; j < poincare_section.cols(); j++){
                file << poincare_section(target_dim - 1, j) << " ";
            }
            file << std::endl;
        }
    }

    file.close();

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


// std::vector<double> loc_max(Eigen::MatrixXcd Mt, int obs_dim, int output_dim){
//     int rowToCopy = obs_dim - 1;
//     std::vector<double> vec(Mt.cols());
//     for (int i = 0; i < Mt.cols(); i++){
//         vec[i] = Mt.cwiseAbs()(rowToCopy, i);
//     }
//     std::vector<double> loc_max_point;
//     loc_max_point.reserve(vec.size()/10000);
//     for (int i = 0; i < vec.size()-6; ++i){
//         if (vec[i+1] - vec[i] > 0
//         && vec[i+2] - vec[i+1] > 0
//         && vec[i+3] - vec[i+2] > 0
//         && vec[i+4] - vec[i+3] < 0
//         && vec[i+5] - vec[i+4] < 0
//         && vec[i+6] - vec[i+5] < 0){
//             loc_max_point.push_back(Mt.cwiseAbs()(output_dim - 1, i+3));
//         }
//     }
//     return loc_max_point;
// }

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

Eigen::MatrixXd poincare_section(Eigen::MatrixXd Mat, int cut_dim, double cut_value){
    // 条件に合えば1, 合わなければ0のベクトルを作成
    std::vector<int> binLoc_max(traj_abs.cols(), 0);

    for (int i = 0; i < traj_abs.cols() -1; ++i){
        if (traj_abs(cut_dim, i) > cut_value && traj_abs(cut_dim, i+1) < cut_value
        || traj_abs(cut_dim, i) < cut_value && traj_abs(cut_dim, i+1) > cut_value){
            binLoc_max[i] = 1;
            binLoc_max[i+1] = 1;
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