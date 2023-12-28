#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <chrono>
#include <omp.h>
#include "cnpy/cnpy.h"
#include "shared/Flow.hpp"
#include "shared/Map.hpp"
#include "shared/myFunc.hpp"
#include "shared/Eigen_numpy_converter.hpp"

int main(){
    auto start = std::chrono::system_clock::now(); // timer start
    SMparams params;
    params.nu = 0.00017;
    params.beta = 0.425;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+4;
    double dump = 1e+3;
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../initials/beta0.4155_nu0.00018_14dim_period.npy", true);
    int numThreads = omp_get_max_threads();
    std::cout << numThreads << "threads" << std::endl;

    int param_steps = 64;
    double beta_begin = 0.416;
    double beta_end = 0.4165;
    double nu_begin = 0.00018;
    double nu_end = 0.00018;
    int loc_max_dim = 3;
    int target_dim = 4;

    ShellModel SM(params, dt, t_0, t, dump, x_0);
    Eigen::VectorXd betas = Eigen::VectorXd::LinSpaced(param_steps, beta_begin, beta_end);
    Eigen::VectorXd nus = Eigen::VectorXd::LinSpaced(param_steps, nu_begin, nu_end);

    std::ostringstream oss;
    oss << "../../bif_data/bif_" << beta_begin <<"to"<< beta_end << "_nu" << nu_begin <<"to" << nu_end << "_" << param_steps << "steps_period" << t-t_0 << "_dump"<< dump << ".txt";  // 文字列を結合する
    std::string fname = oss.str();
    std::ofstream file(fname);
    if (!file) {
        std::cerr << "ファイルを開けませんでした。" << std::endl;
        return 1;
    }
    int counter = 0;
    #pragma omp parallel for num_threads(numThreads) firstprivate(SM, x_0, betas, nus, loc_max_dim, target_dim)
    for (int i = 0; i < param_steps; i++) {
        #pragma omp critical
        std::cout << "\r 現在" << counter << "/" << param_steps << std::flush;
        SM.set_beta_(betas(i));
        SM.nu = nus(i);
        SM.x_0 = myfunc::perturbation(SM.x_0, -4, 0);
        Eigen::MatrixXcd trajectory = SM.get_trajectory();
        PoincareMap map(trajectory.cwiseAbs());
        map.locmax(loc_max_dim-1);
        Eigen::MatrixXd poincare_section = map.get();

        #pragma omp critical
        {
            file << betas(i) << " " << nus(i) << " ";
            //poincare_section_vecを続きに書き込む
            for (int j = 0; j < poincare_section.cols(); j++){
                file << poincare_section(target_dim-1, j) << " ";
            }
            file << std::endl;
        }
        #pragma omp atomic
        counter++;
    }
    file.close();
    myfunc::duration(start);
}