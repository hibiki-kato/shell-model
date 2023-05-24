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
std::vector<double> loc_max(Eigen::MatrixXcd Mt, int obs_dim, int output_dim);


int main(){
    double nu = 0.00017;
    double beta = 0.425;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 100000;
    double latter = 1;
    Eigen::VectorXcd x_0(14);
    x_0(0) = std::complex<double>(0.4350E+00 , 0.5008E+00);
    x_0(1) = std::complex<double>(0.1259E+00 , 0.2437E+00);
    x_0(2) = std::complex<double>(-0.8312E-01 , -0.4802E-01);
    x_0(3) = std::complex<double>(0.5164E-01 , -0.1599E+00);
    x_0(4) = std::complex<double>(-0.1899E+00 , -0.3602E-01);
    x_0(5) = std::complex<double>(0.4093E-03 , 0.8506E-01);
    x_0(6) = std::complex<double>(0.9539E-01 , 0.3215E-01);
    x_0(7) = std::complex<double>(-0.5834E-01 , 0.4433E-01);
    x_0(8) = std::complex<double>(-0.8790E-02 , 0.2502E-01);
    x_0(9) = std::complex<double>(0.3385E-02 , 0.1148E-02);
    x_0(10) = std::complex<double>(-0.7072E-04 , 0.5598E-04);
    x_0(11) = std::complex<double>(-0.5238E-07 , 0.1467E-06);
    x_0(12) = std::complex<double>(0.1E-07 ,0.1E-06);
    x_0(13) = std::complex<double>(0.1E-07 ,0.1E-06);

    int param_steps = 200;
    double beta_begin = 0.415;
    double beta_end = 0.425;
    double nu_begin = 0.00017;
    double nu_begin = 0.00017;
    int loc_max_dim = 3;
    int target_dim = 4;

    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間
    ShellModel solver(nu, beta, f, ddt, t_0, t, latter, x_0);
    Eigen::VectorXd betas = Eigen::VectorXd::LinSpaced(param_steps, beta_begin, beta_end);
    Eigen::VectorXd nue = Eigen::VectorXd::LinSpaced(param_steps, nu_begin, nu_end);
    Eigen::MatrixXcd trajectory;
    std::vector<double> poincare_section;

    // ファイルを開く
    std::ofstream file("data.txt");
    if (!file) {
        std::cerr << "ファイルを開けませんでした。" << std::endl;
        return 1;
    }
    
    for (int i=0; i < param_steps; i++){
        solver.set_beta_(betas(i));
        solver.set_nu_(nues(i));
        trajectory = solver.get_trajectory_();
        poincare_section = loc_max(trajectory);
        file << betas(i) << " " << nues(i) << " ";
        for(const auto& value : poincare_section){
            file << value << " ";
        }
        file << std::endl;
        std::cout << 
    }

    file.close();

    std::
    end = std::chrono::system_clock::now();  // 計測終了時間
    int hours = std::chrono::duration_cast<std::chrono::hours>(end-start).count(); //処理に要した時間を変換
    int minutes = std::chrono::duration_cast<std::chrono::minutes>(end-start).count(); //処理に要した時間を変換
    int seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count(); //処理に要した時間を変換
    int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間を変換
    std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;
}


std::vector<double> loc_max(Eigen::MatrixXcd Mt, int obs_dim, int output_dim){
    int rowToCopy = obs_dim - 1;
    std::vector<double> vec(Mt.cols());
    for (int i = 0; i < Mt.cols(); i++){
        vec[i] = Mt.cwiseAbs()(rowToCopy, i);
    }
    std::vector<double> loc_max_point;
    loc_max_point.reserve(vec.size()/10000);
    for (int i = 0; i < vec.size()-6; ++i){
        if (vec[i+1] - vec[i] > 0
        && vec[i+2] - vec[i+1] > 0
        && vec[i+3] - vec[i+2] > 0
        && vec[i+4] - vec[i+3] < 0
        && vec[i+5] - vec[i+4] < 0
        && vec[i+6] - vec[i+5] < 0){
            loc_max_point.push_back(Mt.cwiseAbs()(output_dim - 1, i+3));
        }
    }
    return loc_max_point;
}