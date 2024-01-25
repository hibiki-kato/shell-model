#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"
namespace plt = matplotlibcpp;


// メイン関数
int main() {
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 0.00018;
    params.beta = 0.417;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 5e+4;
    double dump =1e+4;
    int numThreads = 1;
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../initials/beta0.4163_nu0.00018_10000period_dt0.01.npy", true);
    
    ShellModel SM(params, dt, t_0, t, dump, x_0);
    std::cout << "calculating trajectory" << std::endl;
    // bool laminar = false;
    // Eigen::MatrixXcd rawData = SM.get_trajectory_();
    // データの読み込みをここに記述
    bool laminar = true;
    Eigen::MatrixXcd rawData = npy2EigenMat<std::complex<double>>("../generated_lam/generated_laminar_beta_0.417nu_0.00018_dt0.01_50000period1300check200progresseps0.05.npy", true);

    int dim = rawData.rows() - 1;
    // データの整形(実関数化)
    Eigen::MatrixXd Data(dim*2, rawData.cols());
    for (int i = 0; i < dim; ++i) {
        Data.row(2*i) = rawData.row(i).real();
        Data.row(2*i+1) = rawData.row(i).imag();
    }

    Eigen::VectorXd lyapunovExponents = myfunc::calcLyapunovExponent(SM, Data, numThreads);

    // 結果の表示
    std::cout << lyapunovExponents.rows() << std::endl;
    std::cout << "Lyapunov Exponents:" << std::endl;
    std::cout << lyapunovExponents << std::endl;

    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "18";
    plt::rcparams(plotSettings);
    plt::figure_size(1200, 780);
    // 2からlyapunovExponents.rows()まで等差２の数列

    std::vector<int> xticks(lyapunovExponents.rows());
    std::iota(begin(xticks), end(xticks), 1);
    plt::xticks(xticks);
    // Add graph title
    std::vector<double> x(lyapunovExponents.data(), lyapunovExponents.data() + lyapunovExponents.size());

    plt::plot(xticks, x, "*-");
    // plt::ylim(-1, 1);
    plt::axhline(0, 0, lyapunovExponents.rows(), {{"color", "black"}, {"linestyle", "--"}});
    plt::xlabel("wavenumber");
    plt::ylabel("Lyapunov Exponents");
    std::ostringstream oss;
    if (laminar) {
        oss << "../../lyapunov_exponents/beta" << params.beta << "nu" << params.nu <<"_"<< static_cast<int>(rawData.cwiseAbs().bottomRightCorner(1, 1)(0, 0)) << "period_laminar.png";  // 文字列を結合する
    } else {
        oss << "../../lyapunov_exponents/beta" << params.beta << "nu" << params.nu <<"_"<< static_cast<int>(rawData.cwiseAbs().bottomRightCorner(1, 1)(0, 0)) << "period.png";  // 文字列を結合する
    }
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    // xをテキストファイルに保存
    oss.str("");
    if (laminar) {
        oss << "../../lyapunov_exponents/beta" << params.beta << "nu" << params.nu <<"_dt"<< dt << "_" << static_cast<int>(rawData.cwiseAbs().bottomRightCorner(1, 1)(0, 0)) << "period_laminar.txt";
    } else {
        oss << "../../lyapunov_exponents/beta" << params.beta << "nu" << params.nu <<"_dt"<< dt << "_" << static_cast<int>(rawData.cwiseAbs().bottomRightCorner(1, 1)(0, 0)) << "period.txt";
    }
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "saving as " << fname << std::endl;
    std::ofstream ofs(fname);
    for (int i = 0; i < lyapunovExponents.rows(); ++i) {
        ofs << lyapunovExponents(i) << std::endl;
    }
    ofs.close();

    myfunc::duration(start);
    return 0;
}