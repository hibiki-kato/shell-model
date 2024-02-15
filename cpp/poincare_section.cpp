#include <iostream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <chrono>
#include <omp.h>
#include "cnpy/cnpy.h"
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/Map.hpp"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"
namespace plt = matplotlibcpp;

int main(){
    auto start = std::chrono::system_clock::now(); // timer start
    SMparams params;
    params.nu = 0.00018;
    params.beta = 0.4154;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt =0.01;
    double t_0 = 0;
    double t = 1e+5;
    double dump = 1e+4;
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../initials/beta0.415_nu0.00018_100000period_dt0.01.npy", true);
    std::vector<Eigen::MatrixXd> matrices; //ポアンカレ写像の結果を格納するベクトル
    int plot_dim1 = 3;
    int plot_dim2 = 4;

    ShellModel SM(params, dt, t_0, t, dump, x_0);
    // 計算する場合は以下のコメントアウトを外す
    Eigen::MatrixXcd trajectory = SM.get_trajectory();
    std::string suffix = "";
    // 計算済みの場合は以下のコメントアウトを外す
    // Eigen::MatrixXcd trajectory = npy2EigenMat<std::complex<double>>("../../generated_lam/sync_gen_laminar_beta_0.423nu_0.00018_dt0.01_50000period1000check100progress10^-7-10^-3perturb_4-7_4-10_4-13_7-10_7-13_10-13_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy");
    // std::string suffix = "laminar"

    /*
    ██████         █                                                                     █
    █    █
    █    ██  ███   █  █ ███   ███    ███   ███  ███      █████ ███   ███   █████  █████  █  █ ███   ████
    █    ██ ██  █  █  ██  █  ██  █  █  ██  ██  ██  █     ██  ██  █  █  ██  ██  █  ██  █  █  ██  █  ██  █
    ██████  █   ██ █  █   █  █          █  █   █   █     █   █   ██     █  █   ██ █   ██ █  █   █  █   █
    █       █    █ █  █   █  █       ████  █   █████     █   █   ██  ████  █   ██ █   ██ █  █   █  █   █
    █       █   ██ █  █   █  █      █   █  █   █         █   █   ██ █   █  █   ██ █   ██ █  █   █  █   █
    █       ██  █  █  █   █  ██  █  █  ██  █   ██  █     █   █   ██ █  ██  ██  █  ██  █  █  █   █  ██  █
    █        ███   █  █   █   ███   █████  █    ████     █   █   ██ █████  █████  █████  █  █   █   ████
                                                                           █      █                    █
                                                                           █      █                █   █
                                                                           █      █                 ███
    */

    PoincareMap PM(trajectory.cwiseAbs());
    PM.locmax(3);
    // PM.poincare_section(6, 1);
    std::cout << PM.indices.size() << std::endl;
    std::cout << PM.indices[0].size() << std::endl;
    Eigen::MatrixXd result = PM.get();

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
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    plt::figure_size(1200, 1200);
    Eigen::VectorXd vec1 = result.row(plot_dim1-1);
    Eigen::VectorXd vec2 = result.row(plot_dim2-1);
    std::vector<double> x(vec1.data(), vec1.data() + vec1.size());
    std::vector<double> y(vec2.data(), vec2.data() + vec2.size());
    
    plt::scatter(x, y, 5.0);
    std::ostringstream oss;
    plt::xlabel(myfunc::ordinal_suffix(plot_dim1) + " Shell"); 
    plt::ylabel(myfunc::ordinal_suffix(plot_dim2) + " Shell");
    // plt::ylim(0.15, 0.4);
    // plt::xlim(0.05, 0.6);
    oss.str("");
    oss << "../../poincare/beta" << params.beta << "nu" << params.nu << "t" << t - t_0  << suffix << ".png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    myfunc::duration(start);
}