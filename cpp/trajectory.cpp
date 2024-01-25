#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <chrono>
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"
namespace plt = matplotlibcpp;

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 4e-5;
    params.beta = 0.5;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+3;
    double dump = 1e+5;
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../initials/beta0.5_nu4e-05_15dim.npy", true);
    int plotdim1 = 4;
    int plotdim2 = 5;

    ShellModel SM(params, dt, t_0, t, dump, x_0);
    Eigen::MatrixXcd trajectory = SM.get_trajectory(); 
    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "20";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1200, 1200);
    // Add graph title
    std::vector<double> x(trajectory.cols()),y(trajectory.cols());

    for(int i=0;i<trajectory.cols();i++){
        x[i]=std::abs(trajectory(plotdim1 - 1, i));
        y[i]=std::abs(trajectory(plotdim2 - 1, i));
    }
    std::map<std::string, std::string> keywords;
    keywords["lw"] = "1";   
    plt::xlim(0.0, 0.45);
    plt::ylim(0.0, 0.45);
    plt::xlabel("$|U_{" + std::to_string(plotdim1) + "}|$");
    plt::ylabel("$|U_{" + std::to_string(plotdim2) + "}|$");
    plt::plot(x,y, keywords);
    std::ostringstream oss;
    oss << "../../traj_img/beta_" << params.beta << "nu_" << params.nu <<"_"<< static_cast<int>(t-t_0) << "period.png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    oss.str("");
    //  文字列を取得する
    oss << "../../traj/beta" << params.beta << "_nu" << params.nu <<"_"<< static_cast<int>(t-t_0) << "period.npy";  // 文字列を結合する
    std::string npyfname = oss.str();
    std::cout << "Saving result to " << npyfname << std::endl;
    // EigenMat2npy(trajectory, npyfname);

    myfunc::duration(start);
}