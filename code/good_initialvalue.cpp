#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include "Runge_Kutta.hpp"
#include <chrono>
#include <random>
#include <omp.h>
#include "matplotlibcpp.h"
#include "cnpy/cnpy.h"
namespace plt = matplotlibcpp;
Eigen::VectorXcd npy2EigenVec(const char* fname);
void EigenVec2npy(Eigen::VectorXcd Vec, std::string fname);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    
    // generating laminar sample !DO NOT CHANGE!
    double nu = 0.00018;
    double beta = 0.41616;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.001;
    double t_0 = 0;
    double t = 1000;
    double latter = 4;
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.41616nu0.00018_1.05286e+07period.npy");
    ShellModel SM(nu, beta, f, ddt, t_0, t, latter, x_0);
    Eigen::MatrixXcd laminar = SM.get_trajectory_();
    int numRows = laminar.cols() / 10;
    Eigen::MatrixXcd laminar_sample(laminar.rows(), numRows);
    for (int i = 0; i < numRows; i++){
        int colIdx = 10 * i;
        laminar_sample.col(i) = laminar.col(colIdx);
    }

    // set up for search
    t=3000;
    latter = 1;
    nu = 0.00018;
    beta = 0.417;
    x_0 = npy2EigenVec("../../initials/beta0.417_nu0.00018_2000period.npy");
    int num_of_candidates = 32;
    int skip = 100;
    double epsilon = 2E-2;
    int threads = omp_get_max_threads();
    std::cout << threads << "threads" << std::endl;

    LongLaminar LL(nu, beta, f, ddt, t_0, t, latter, x_0, laminar_sample, epsilon, skip, 100, 10, threads);
    Eigen::MatrixXcd initials(x_0.size(), num_of_candidates);
    double longest;

    for(int i = 0; i < 40; i++){
        // make matrix that each cols are candidates of initial value
        std::cout << "現在"  << i+1 << "回" <<std::endl;
        initials.col(0) = LL.get_x_0_();
        for(int j = 1; j < num_of_candidates - 1; j++){
            initials.col(j) = LL.perturbation_(LL.get_x_0_(), -10, -1);
        }
        Eigen::VectorXd durations(num_of_candidates);
        #pragma omp parallel for num_threads(threads)
        for(int j = 0; j < num_of_candidates; j++){
            if (omp_get_thread_num() == 0){
                std::cout << "\r" << (j + 1) * threads << "個目" << std::flush;
            }
            LongLaminar local_LL = LL;
            local_LL.set_x_0_(initials.col(j));
            Eigen::MatrixXcd trajectory = local_LL.get_trajectory_();
            durations(j) = local_LL.laminar_persistent_(trajectory);
            }
        int maxId;
        longest = durations.maxCoeff(&maxId);
        LL.set_x_0_(initials.col(maxId));
        std::cout << "現在最高" << longest << std::endl;
        if (longest > 0.99*t){
            break;
        }
    }
    
    std::ostringstream oss;
    oss << "../../initials/beta" << beta << "_nu" << nu<< "_" << static_cast<int>(longest+0.5) << "period.npy";  // 文字列を結合する
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "saving as " << fname << std::endl;
    EigenVec2npy(LL.get_x_0_(), fname);

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

void EigenVec2npy(Eigen::VectorXcd Vec, std::string fname){
    std::vector<std::complex<double>> x(Vec.size());
    for(int i=0;i<Vec.size();i++){
        x[i]=Vec(i);
    }
    cnpy::npy_save(fname, &x[0], {(size_t)Vec.size()}, "w");
}