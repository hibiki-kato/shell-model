                                                      █                                     █
█████                            █                    █     ██                              █         ██
█    █                                                █     ██                              █         ██
█    █   ███   █████ ███   ███   █  █ ███   ███    ████    ████  ███      █   █  █████   ████   ███  ████  ███
█    █  ██  █  ██  ██  █  █  ██  █  ██  █  ██  █  ██  █     ██  ██  █     █   █  ██  █  ██  █  █  ██  ██  ██  █
█████   █   █  █   █   ██     █  █  █   █  █   █  █   █     ██  █   ██    █   █  █   ██ █   █      █  ██  █   █
█   █   █████  █   █   ██  ████  █  █   █  █████  █   █     ██  █    █    █   █  █   ██ █   █   ████  ██  █████
█   ██  █      █   █   ██ █   █  █  █   █  █      █   █     ██  █   ██    █   █  █   ██ █   █  █   █  ██  █
█    █  ██  █  █   █   ██ █  ██  █  █   █  ██  █  ██  █     ██  ██  █     █   █  ██  █  ██  █  █  ██  ██  ██  █
█    ██  ████  █   █   ██ █████  █  █   █   ████   ████      ██  ███       ████  █████   ████  █████   ██  ████
                                                                                 █
                                                                                 █
                                                                                 █
/**
 * @file stagger_and_step_sync.cpp
 * @author Hibiki Kato
 * @brief Conduct stagger and step method using synchronization
 * @version 0.1
 * @date 2023-09-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <utility> // std::pair用
#include <tuple>
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include <chrono>
#include "cnpy/cnpy.h"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"

namespace plt = matplotlibcpp;
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXcd> calc_next(ShellModel&, Eigen::VectorXd pre_n, Eigen::VectorXd pre_theta, Eigen::VectorXcd previous);
bool isLaminar(Eigen::VectorXd phases, std::vector<std::tuple<int, int, double>> sync_pairs);
int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 0.00018;
    params.beta = 0.417;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 30000;
    double t = 50000;
    double dump = 1;
    double check = 6000;
    double progress = 500;
    int perturb_min = -13;
    int perturb_max = -5;
    int limit = 1e+5; //limitation of trial of stagger and step
    Eigen::MatrixXcd loaded = npy2EigenMat<std::complex<double>>("../../generated_lam/sync_gen_laminar_beta_0.417nu_0.00018_dt0.01_53000period5000check500progress10^-14-10^-5perturb_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy");
    Eigen::VectorXcd x_0 = loaded.block(0, t_0*100, 14, 1);
    // Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../../initials/beta0.417_nu0.00018_9-12.npy");
    ShellModel SM(params, dt, t_0, t, latter, x_0);
    Eigen::MatrixXcd Dummy_Laminar(x_0.rows()+1, 1); //dummy matrix to use LongLaminar Class
    LongLaminar LL(nu, beta, f, dt, t_0, t, latter, x_0, Dummy_Laminar, 0.01, 100, check, progress, 8);
    int numThreads = omp_get_max_threads();

    //make pairs of shells to observe phase difference(num begins from 1)
    std::vector<std::tuple<int, int, double>> sync_pairs;
    // sync_pairs.push_back(std::make_tuple(4, 7, 2.3));
    // sync_pairs.push_back(std::make_tuple(4, 10, 2.3));
    // sync_pairs.push_back(std::make_tuple(4, 13, 2.3));
    // sync_pairs.push_back(std::make_tuple(7, 10, 2));
    // sync_pairs.push_back(std::make_tuple(7, 13, 2));
    // sync_pairs.push_back(std::make_tuple(10, 13, 1E-1));

    sync_pairs.push_back(std::make_tuple(5, 8, 2.3));
    sync_pairs.push_back(std::make_tuple(5, 11, 2.3));
    sync_pairs.push_back(std::make_tuple(5, 14, 2.3));
    sync_pairs.push_back(std::make_tuple(8, 11, 0.7));
    sync_pairs.push_back(std::make_tuple(8, 14, 0.7));
    sync_pairs.push_back(std::make_tuple(11, 14, 1E-1));

    sync_pairs.push_back(std::make_tuple(6, 9, 2.3));
    sync_pairs.push_back(std::make_tuple(6, 12, 2.3));
    sync_pairs.push_back(std::make_tuple(9, 12, 0.3));

    // sync_pairs.push_back(std::make_tuple(9, 12, 3.2)); // dummy to check all trajectory

    /*
      ██                                                                    █
    █████                                                                   █
    █      ██                                                               █            ██
    █     ████   ████   █████   █████   ████   █ ██      ████  ██████   █████     █████ ████   ████   █████
    ███    █        █  ██  ██  ██  ██  █   ██  ██           █  ██   █  ██  ██     █      █    █   ██  ██   █
      ███  █        █  █    █  █    █  █   ██  █            █  █    █  █    █     ██     █    █   ██  █    █
        ██ █    █████  █    █  █    █  ██████  █        █████  █    █  █    █      ███   █    ██████  █    █
        ██ █    █   █  █    █  █    █  █       █        █   █  █    █  █    █        ██  █    █       █    █
        █  ██   █   █  ██  ██  ██  ██  ██      █         █   █  █    █  ██  ██        ██  ██   ██      ██   █
    █████   ███ █████   █████   █████   ████   █        █████  █    █   ███ █     ████    ███  ████   █████
                            █       █                                                                █
                           ██      ██                                                                █
                       █████   █████                                                                 █
    */
    Eigen::MatrixXcd calced_laminar(x_0.rows()+1, SM.get_steps_()+1);
    int stagger_and_step_num = static_cast<int>((t-t_0) / progress + 0.5);
    int check_steps = static_cast<int>(check / dt + 0.5);
    int progress_steps = static_cast<int>(progress / dt + 0.5);
    SM.set_steps_(check_steps);
    Eigen::VectorXd n = Eigen::VectorXd::Zero(x_0.rows());

    Eigen::VectorXd next_n(x_0.rows()); // candidate of next n
    double max_perturbation = 0; // max perturbation
    double min_perturbation = 1; // min perturbation

    for (int i; i < stagger_and_step_num; i++){
        std::cout << "\r 現在" << SM.get_t_0_() << "時間" << std::flush;
        bool laminar = true; // flag
        double now_time = SM.get_t_0_();
        
        Eigen::VectorXcd now = SM.get_x_0_();
        Eigen::MatrixXcd trajectory = Eigen::MatrixXcd::Zero(x_0.rows()+1, progress_steps+1); //wide matrix for progress
        trajectory.topLeftCorner(x_0.size(), 1) = now;
        trajectory(now.rows(), 0) = now_time;
        Eigen::VectorXd theta = now.cwiseArg();
        Eigen::VectorXd start_n = n; // preserve n at the start of the loop for stagger and step
         double max_duration = check - check; // max duration of laminar
        // calculate rotation number
        // no perturbation at first
        for (int j = 0; j < check_steps; j++){
            std::tie(n, theta, now) = calc_next(SM, n, theta, now);
            if (isLaminar(theta+n*2*M_PI, sync_pairs)){
                if (j < progress_steps){
                    now_time += dt;
                    trajectory.block(0, j+1, x_0.rows(), 1) = now;
                    trajectory(x_0.rows(), j+1) = now_time;
                }
                if (j+1 == progress_steps){
                    next_n = n; //preserve candidate of n
                }
            }
            else{
                laminar = false;
                max_duration += j*dt;
                break;
            }
        }
        // if laminar, continue to for loop
        if (laminar){
            SM.set_t_0_(trajectory.bottomRightCorner(1, 1).cwiseAbs()(0, 0));
            SM.set_x_0_(trajectory.topRightCorner(now.rows(), 1));
            calced_laminar.middleCols(i*progress_steps, progress_steps+1) = trajectory;
            n = next_n;
            continue;
        }
        // otherwise, try stagger and step in parallel
        else{
            /*
             ███    ██    ███     ███ ██████   ██   ████  ██████
            █       ██   █       █       █     ██   █   █    █
            █      █ █   █       █       █    █ █   █   █    █
            ███    █  █  ███     ███     █    █  █  █   █    █
              ██  ██  █    ██      ██    █   ██  █  ████     █
               █  ██████    █       █    █   ██████ █  █     █
               █  █    █    █       █    █   █    █ █  ██    █
            ███  █     █ ███     ███     █  █     █ █   ██   █
            */
            std::cout << std::endl;
            int counter = 0;
            bool success = false; // whether stagger and step succeeded
            Eigen::VectorXcd original_x_0 = SM.get_x_0_(); // log original x_0 for calc overall perturbation
            // parallelization doesn't work well without option
            #pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(success, max_duration, SM, n, counter, max_perturbation, original_x_0) firstprivate(LL, sync_pairs, check_steps, progress_steps, numThreads, start_n)
            for (int j = 0; j < limit; j++){
                if (success){
                    continue;
                }
                // show progress
                if (omp_get_thread_num() == 0)
                {
                    counter++;
                    std::cout << "\r " << counter * numThreads << "試行　最高" << max_duration << "/"<< check << std::flush;
                }
                bool Local_laminar = true; // flag
                LongLaminar Local_LL = LL; // copy of LL
                ShellModel Local_SM = SM; // copy of SM
                std::vector<std::tuple<int, int, double>> Local_sync_pairs = sync_pairs; // copy of sync_pairs
                double Local_now_time = Local_SM.get_t_0_();
                Eigen::VectorXcd Local_x_0 = Local_LL.perturbation_(Local_SM.get_x_0_(), perturb_min, perturb_max); // perturbed initial state
                Eigen::VectorXcd Local_now = Local_x_0; // perturbed initial state
                Eigen::MatrixXcd Local_trajectory = Eigen::MatrixXcd::Zero(Local_now.rows()+1, progress_steps+1); //wide matrix for progress
                Local_trajectory.topLeftCorner(Local_now.rows(), 1) = Local_now;
                Local_trajectory(Local_now.rows(), 0) = Local_now_time;
                Eigen::VectorXd Local_theta = Local_now.cwiseArg();
                Eigen::VectorXd Local_n = start_n; // It doesn't matter if this is not accurate due to the perturbation, because wrong case won't survive.
                Eigen::VectorXd Local_next_n;
                for (int k = 0; k < check_steps; k++){
                    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXcd> Local_next = calc_next(Local_SM, Local_n, Local_theta, Local_now);
                    Local_n = std::get<0>(Local_next);
                    Local_theta = std::get<1>(Local_next);
                    Local_now = std::get<2>(Local_next);
                    Local_now_time += Local_SM.get_ddt_();
                    if (isLaminar(Local_theta+Local_n*2*M_PI, Local_sync_pairs)){
                        if (k < progress_steps){
                            Local_trajectory.block(0, k+1, Local_now.rows(), 1) = Local_now;
                            Local_trajectory(Local_now.rows(), k+1) = Local_now_time;
                        }
                        if (k+1 == progress_steps){
                            Local_next_n = Local_n; //preserve candidate of n
                        }
                    }
                    else{
                        #pragma omp critical
                        if (Local_now_time - Local_SM.get_t_0_() > max_duration && success == false){
                            {
                                max_duration = Local_now_time - Local_SM.get_t_0_();
                                // (SM.get_x_0_() - Local_x_0).norm(); //perturbation size
                                SM.set_x_0_(Local_x_0);
                            }
                        }
                        Local_laminar = false;
                        break;
                    }
                }
                /*
                 ████  █    █   ████   ████   ██████  ████   ████
                █  ██  █    █  ██  ██ ██  ██  █      █  ██  ██  ██
                █   █  █    █ ██    █ █    █  █      █   █  █    █
                ███    █    █ ██      █       █ ███  ███    ███
                  ███  █    █ ██      █       █ ███    ███    ███
                █   ██ █    █ ██    █ █    ██ █     ██   ██ █    █
                █   ██ ██  ██  █   ██ ██   █  █      █   ██ █   ██
                █████   ████   █████   ████   ██████ █████  █████
                */
                #pragma omp critical
                if (Local_laminar == true && success == false){
                    {   
                        double perturbation_size = (Local_trajectory.topLeftCorner(Local_now.rows(), 1) - original_x_0).norm();
                        if (perturbation_size > max_perturbation){
                            max_perturbation = perturbation_size;
                        }
                        if (perturbation_size < min_perturbation){
                            min_perturbation = perturbation_size;
                        }
                        std::cout << "overall perturbation scale here is " << perturbation_size << std::endl;
                        SM.set_t_0_(Local_trajectory.bottomRightCorner(1, 1).cwiseAbs()(0, 0));
                        SM.set_x_0_(Local_trajectory.topRightCorner(Local_now.rows(), 1));
                        calced_laminar.middleCols(i*progress_steps, progress_steps+1) = Local_trajectory;
                        n = Local_next_n;
                        success = true;
                    }
                }
            } // end of stagger and step for loop

            if (!success){
                std::cout << "stagger and step failed" << std::endl;
                // 成功した分だけcalced_laminarをresize
                calced_laminar.conservativeResize(x_0.rows()+1, i*progress_steps+1);
                break;
            }
        }// end of stagger and step
    }

    // order of max perturbation
    std::cout << "max perturbation is " << max_perturbation << std::endl;
    std::cout << "min perturbation is " << min_perturbation << std::endl;
    int logged_max_perturbation = static_cast<int>(log10(max_perturbation)-0.5) - 1;
    std::cout << "logged max perturbation is " << logged_max_perturbation << std::endl;
    int logged_min_perturbation = static_cast<int>(log10(min_perturbation) - 0.5) -1;
    std::cout << "logged min perturbation is " << logged_min_perturbation << std::endl;

    /*
            █             
    █████   █          █
    ██  ██  █          █
    ██   █  █   ████  ████
    ██  ██  █  ██  ██  █  
    █████   █  █    █  █  
    ██      █  █    █  █  
    ██      █  █    █  █  
    ██      █  ██  ██  ██ 
    ██      █   ████    ██
    */
    std::cout << "plotting" << std::endl;
    // plot settings
    int skip = 1; // plot every skip points
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1200, 1200);
    std::vector<double> x(calced_laminar.cols()),y(calced_laminar.cols());
    int reach = static_cast<int>(calced_laminar.bottomRightCorner(1, 1).cwiseAbs()(0, 0) + 0.5); 

    for(int i=0;i<calced_laminar.cols();i++){
        x[i]=calced_laminar.cwiseAbs()(3, i);
        y[i]=calced_laminar.cwiseAbs()(4, i);
    }
    plt::xlim(0.0, 0.4);
    plt::ylim(0.0, 0.4);
    plt::plot(x,y);
    // 最後の点を赤でプロット(サイズは大きめ)
    std::vector<double> x_last(1), y_last(1);
    x_last[0] = x[x.size()-1];
    y_last[0] = y[y.size()-1];
    std::map<std::string, std::string> lastPointSettings;
    lastPointSettings["color"] = "red";
    lastPointSettings["marker"] = "o";
    lastPointSettings["markersize"] = "5";
    plt::plot(x_last, y_last, lastPointSettings);
    // 最初の点を緑でプロット(サイズは大きめ)
    std::vector<double> x_first(1), y_first(1);
    x_first[0] = x[0];
    y_first[0] = y[0];
    std::map<std::string, std::string> firstPointSettings;
    firstPointSettings["color"] = "green";
    firstPointSettings["marker"] = "o";
    firstPointSettings["markersize"] = "5";
    plt::plot(x_first, y_first, firstPointSettings);

    std::ostringstream oss;
    oss << "../../generated_lam_imag/sync_gen_laminar_beta" << params.beta << "nu" << params.nu <<"_dt"<< dt << "_" << reach << "period" << check << "check" << progress << "progress10^" << logged_min_perturbation<<"-10^"<< logged_max_perturbation << "perturb";
    for (const auto& pair : sync_pairs){
        oss << "_" << std::get<0>(pair) << "-" << std::get<1>(pair);
    }
    oss << ".png";
    std::string filename = oss.str(); // 文字列を取得する
    std::cout << "\n Saving result to " << filename << std::endl;
    plt::save(filename);

    /*
      ██████
     ███ ███
    ██                                             █
    ██                                             █
    ██          █████   ██      █   █████        ██████    █████          █ █████     █ █████   ██      █
     ██        ██  ███   █     ██  ██   ██         ██     ███  ██         ███  ███    ███  ███   █     ██
      ███           ██   ██    ██  █     █         █     ██     ██        ██    ██    ██    ██   ██    ██
        ███         ██   ██   ██  ██     ██        █     ██      █        █     ██    █      █   ██    █
          ██    ██████    █   ██  █████████        █     ██      █        █      █    █      ██   █   ██
           ██ ███   ██    ██  █   ██               █     ██      █        █      █    █      ██   ██  ██
           ██ ██    ██    ██ ██   ██               █     ██      █        █      █    █      █     █  █
           █  ██    ██     █ ██   ██               █      █     ██        █      █    ██    ██     █ ██
    ███  ███  ██   ███     ███     ███  ██         ██     ███  ██         █      █    ███  ███     ███
    ██████     ████  █      ██      ██████          ███    █████          █      █    █ █████       ██
                                                                                      █             ██
                                                                                      █             █
                                                                                      █            ██
                                                                                      █           ██
                                                                                      █         ███
    */

    oss.str("");
    if (progress == t){
        oss << "../../initials/beta" << beta << "_nu" << nu<< "_" << check << "period_dt" << dt;
        for (const auto& pair : sync_pairs){
            oss << "_" << std::get<0>(pair) << "-" << std::get<1>(pair);
        }
        oss << ".npy";
        std::string fname = oss.str(); // 文字列を取得する
        std::cout << "saving as " << fname << std::endl;
        EigenVec2npy(calced_laminar.topLeftCorner(calced_laminar.rows()-1, 1).col(0), fname);
    } else{
        oss << "../../generated_lam/sync_gen_laminar_beta" << params.beta << "nu" << params.nu <<"_dt"<< dt << "_" << reach << "period" << check << "check" << progress << "progress10^" << logged_min_perturbation<<"-10^"<< logged_max_perturbation << "perturb";
         for (const auto& pair : sync_pairs){
            oss << "_" << std::get<0>(pair) << "-" << std::get<1>(pair);
        }
        oss << ".npy";
        std::string fname = oss.str(); // 文字列を取得する
        std::cout << "saving as " << fname << std::endl;
        EigenMat2npy(calced_laminar, fname);
    }

    myfunc::duration(start);
}

bool isLaminar(Eigen::VectorXd phases, std::vector<std::tuple<int, int, double>> sync_pairs){
    bool allSync = true; // flag 
    for (const auto& pair : sync_pairs){
        // if any pair is not sync, allSync is false
        double phase_diff = std::abs(phases(std::get<0>(pair)-1) - phases(std::get<1>(pair)-1));
        double threshold = std::get<2>(pair);
        if((threshold < phase_diff && phase_diff < 2* M_PI - threshold) || (2* M_PI + threshold < phase_diff)){
            allSync = false;
            break;
        }
    }
    return allSync;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXcd> calc_next(ShellModel& SM, Eigen::VectorXd pre_n, Eigen::VectorXd pre_theta, Eigen::VectorXcd previous){
    Eigen::VectorXcd now = SM.rk4(previous);
    Eigen::VectorXd theta = now.cwiseArg();
    Eigen::VectorXd n = pre_n;
    for(int i; i < theta.size(); i++){
        n(i) = shift(pre_theta(i), theta(i), pre_n(i));
    }
    return std::make_tuple(n, theta, now);
}
