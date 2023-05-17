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

int main(){
    double nu = 0.00017256;
    double beta = 0.417;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 1000;
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

    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間
    ShellModel solver(nu, beta, f, ddt, t_0, t, latter, x_0);

    Eigen::MatrixXcd trajectory = solver.get_trajectory_();
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1200, 780);
    // Add graph title
    plt::title("Sample figure");
    std::vector<double> x(trajectory.cols()),y(trajectory.cols());

    for(int i=0;i<trajectory.cols();i++){
        x[i]=trajectory.cwiseAbs()(14, i);
        y[i]=trajectory.cwiseAbs()(0, i);
    }

    plt::plot(x,y);
    std::ostringstream oss;
    oss << "../beta_" << beta << "nu_" << nu <<"_"<< t-t_0 << "period.png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    oss.str(""); //initializing the stringstream
    Eigen::MatrixXcd transposed = trajectory.transpose();
    // map to const mats in memory
    Eigen::Map<const Eigen::MatrixXcd> MOut(&transposed(0,0), transposed.rows(), transposed.cols());
    oss << "../beta_" << beta << "nu_" << nu <<"_"<< t-t_0 << "period.npy";  // 文字列を結合する
    std::string npzfname = oss.str(); // 文字列を取得する

    
    // save to np-arrays files
    cnpy::npy_save(npzfname, MOut.data(), {(size_t)transposed.rows(), (size_t)transposed.cols()}, "w");

    end = std::chrono::system_clock::now();  // 計測終了時間
    int hours = std::chrono::duration_cast<std::chrono::hours>(end-start).count(); //処理に要した時間を変換
    int minutes = std::chrono::duration_cast<std::chrono::minutes>(end-start).count(); //処理に要した時間を変換
    int seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count(); //処理に要した時間を変換
    int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間を変換
    std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;

    // std::ofstream file("output.txt");
    // file << trajectory.cwiseAbs();
    // file.close();


}