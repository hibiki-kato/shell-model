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

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    // generating laminar sample for detection
    // !DO NOT CHANGE!
    double nu = 0.00017520319481270297;
    double beta = 0.416;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 50000;
    double latter = 200;
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.416_nu0.00017520319481270297_step0.01_10000.0period_laminar.npy");

    double epsilon=1E-1;
    int skip = 10000;
    double check_sec = 2000;
    double progress_sec = 400;
    int threads = omp_get_max_threads();

    ShellModel SM(nu, beta, f, ddt, t_0, t, latter, x_0);
    Eigen::MatrixXcd laminar = SM.get_trajectory_();
    int numCols = laminar.cols() / 10;
    Eigen::MatrixXcd laminar_sample(laminar.rows(), numCols);
    for (int i = 0; i < numCols; i++){
        int colIdx = 10 * i;
        laminar_sample.col(i) = laminar.col(colIdx);
    }

    beta = 4.16159e-01;
    std::cout << "beta = " << beta << std::endl;
    nu = 0.00018;
    latter = 1;
    double dump = 1e+4;
    t = 1e+9;
    t_0 = 0;
    x_0 = npy2EigenVec("../../initials/beta0.418_nu0.00018_4000period.npy");

    Eigen::VectorXcd x = x_0;

    LongLaminar LL(nu, beta, f, ddt, t_0, t, latter, x_0, laminar_sample, epsilon, skip, check_sec, progress_sec, threads);
    LongLaminar LL_for_dump = LL;
    LL_for_dump.set_t_(dump);
    LL_for_dump.set_x_0_(LL_for_dump.perturbator_(LL_for_dump.get_x_0_()));
    LL.set_x_0_(LL_for_dump.get_trajectory_().topRightCorner(14, 1));
    std::vector<double> durations = LL.laminar_duration_();

    Eigen::VectorXd durations_vec = Eigen::Map<Eigen::VectorXd>(&durations[0], durations.size());
    std::ostringstream oss;
    oss << "../../distribution/beta" << beta << "nu_" << nu <<"_"<< t << "period.npy";  // 文字列を結合する
    std::string filename = oss.str(); // 文字列を取得する
    std::cout << "\n Saving result to " << filename << std::endl;
    EigenVec2npy(durations_vec, filename);
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

void EigenVec2npy(Eigen::VectorXd Vec, std::string fname){
    std::vector<double> x(Vec.size());
    for(int i=0;i<Vec.size();i++){
        x[i]=Vec(i);
    }
    cnpy::npy_save(fname, &x[0], {(size_t)Vec.size()}, "w");
}


    