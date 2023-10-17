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
#include "Runge_Kutta.hpp"
#include <chrono>
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname);
Eigen::VectorXcd npy2EigenVec(const char* fname);
int shift(double pre_theta, double theta, int rotation_number);
bool isLaminar(Eigen::VectorXd phases, std::vector<std::tuple<int, int, double>> sync_pairs);
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXcd> calc_next(ShellModel&, Eigen::VectorXd pre_n, Eigen::VectorXd pre_theta, Eigen::VectorXcd previous);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double nu = 0.00018;
    double beta = 0.417;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 2e+4;
    double latter = 1;
    double check = 1000;
    double progress = 100;
    int limit = 1e+5; //limitation of trial of stagger and step
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.417_nu0.00018_13348period_dt0.01eps0.005.npy");
    ShellModel SM(nu, beta, f, dt, t_0, t, latter, x_0);
    Eigen::MatrixXcd Dummy_Laminar(x_0.rows()+1, 1); //dummy matrix to use LongLaminar Class
    LongLaminar LL(nu, beta, f, dt, t_0, t, latter, x_0, Dummy_Laminar, 0.01, 100, check, progress, 8);
    int numThreads = omp_get_max_threads();

    //make pairs of shells to observe phase difference(num begins from 1)
    std::vector<std::tuple<int, int, double>> sync_pairs;
    // sync_pairs.push_back(std::make_tuple(4, 7, 2));
    // sync_pairs.push_back(std::make_tuple(4, 10, 2));
    // sync_pairs.push_back(std::make_tuple(4, 13, 2));
    // sync_pairs.push_back(std::make_tuple(7, 10, 2));
    // sync_pairs.push_back(std::make_tuple(7, 13, 1.1));
    // sync_pairs.push_back(std::make_tuple(10, 13, 3.3E-2));

    // sync_pairs.push_back(std::make_tuple(5, 8, 2));
    // sync_pairs.push_back(std::make_tuple(5, 11, 2));
    // sync_pairs.push_back(std::make_tuple(5, 14, 2));
    // sync_pairs.push_back(std::make_tuple(8, 11, 0.55));
    // sync_pairs.push_back(std::make_tuple(8, 14, 0.55));
    // sync_pairs.push_back(std::make_tuple(11, 14, 8E-3));

    // sync_pairs.push_back(std::make_tuple(6, 9, 1.7));
    // sync_pairs.push_back(std::make_tuple(6, 12, 1.7));
    sync_pairs.push_back(std::make_tuple(9, 12, 0.2));

    // sync_pairs.push_back(std::make_tuple(9, 12, 0.06)); // dummy to check all trajectory

    /*
      ██                                                                    █
    █████                                                                   █
    █      ██                                                               █            ██
    █     ████   ████   █████   █████   ████   █ ██      ████  ██████   █████     █████ ████   ████   █████
    ███    █        █  ██  ██  ██  ██  █   ██  ██           █  ██   █  ██  ██     █      █    █   ██  ██   █
      ███  █        █  █    █  █    █  █   ██  █            █  █    █  █    █     ██     █    █   ██  █    █
        ██ █    █████  █    █  █    █  ██████  █        █████  █    █  █    █      ███   █    ██████  █    █
        ██ █    █   █  █    █  █    █  █       █        █   █  █    █  █    █        ██  █    █       █    █
        █  ██   █   █  ██  ██  ██  ██  ██      █        █   █  █    █  ██  ██        ██  ██   ██      ██   █
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

    for (int i; i < stagger_and_step_num; i++){
        std::cout << "\r 現在" << SM.get_t_0_() << "時間" << std::flush;
        bool laminar = true; // flag
        double now_time = SM.get_t_0_();
        Eigen::VectorXcd now = SM.get_x_0_();
        Eigen::MatrixXcd trajectory = Eigen::MatrixXcd::Zero(x_0.rows()+1, progress_steps+1); //wide matrix for progress
        trajectory.topLeftCorner(x_0.size(), 1) = now;
        trajectory(now.rows(), 0) = now_time;
        Eigen::VectorXd theta = now.cwiseArg();
        // calculate rotation number
        // noperturbation at first
        for (int j = 0; j < check_steps; j++){
            std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXcd> next = calc_next(SM, n, theta, now);
            n = std::get<0>(next);
            theta = std::get<1>(next);
            now = std::get<2>(next);
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
        // otherwise, try stagger and step parallelly
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
            double max_duration = check - progress; // max duration of laminar
            double total_perturbation = 0; // total perturbation
            // successとmax_durationとcounter以外はprivateにする
            #pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(success, max_duration, SM, n, total_perturbation, counter) firstprivate(LL, sync_pairs, check_steps, progress_steps, numThreads)
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
                Eigen::VectorXcd Local_x_0 = Local_LL.perturbation_(Local_SM.get_x_0_(), -16, -6);
                Eigen::VectorXcd Local_now = Local_x_0; // perturbed initial state
                Eigen::MatrixXcd Local_trajectory = Eigen::MatrixXcd::Zero(Local_now.rows()+1, progress_steps+1); //wide matrix for progress
                Local_trajectory.topLeftCorner(Local_now.rows(), 1) = Local_now;
                Eigen::VectorXd Local_theta = Local_now.cwiseArg();
                Eigen::VectorXd Local_n = n; // It doesn't matter if this is not accurate because of perturbation, because it won't survive.
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
                                total_perturbation += (SM.get_x_0_() - Local_x_0).norm();
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
                        std::cout << "overall perturbation scale here is " << total_perturbation + (Local_trajectory.topLeftCorner(Local_now.rows(), 1) - SM.get_x_0_()).norm() << std::endl;
                        SM.set_t_0_(Local_trajectory.bottomRightCorner(1, 1).cwiseAbs()(0, 0));
                        std::cout << "更新後の時間は" << SM.get_t_0_() << std::endl;
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
                if (max_duration-SM.get_t_0_() <= 0.1){
                    std::cout << "nは" << n << std::endl;
                    std::cout << "thetaは" << SM.get_x_0_().cwiseArg() << std::endl;
                    std::cout << "角度は" << theta + 2*n*M_PI << std::endl;
                }
                break;
            }
        }// end of stagger and step
    }


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
    std::ostringstream oss;
    oss << "../../generated_lam_imag/sync_gen_laminar_beta_" << beta << "nu_" << nu <<"_dt"<< dt << "_" << reach << "period" << check << "check" << progress << "progress";
    for (const auto& pair : sync_pairs){
        oss << "_" << std::get<0>(pair) << "-" << std::get<1>(pair);
    }
    oss << ".png";
    std::string filename = oss.str(); // 文字列を取得する
    std::cout << "\n Saving result to " << filename << std::endl;
    plt::save(filename);

    oss.str("");
    oss << "../../generated_lam/sync_gen_laminar_beta_" << beta << "nu_" << nu <<"_dt"<< dt << "_" << reach << "period" << check << "check" << progress << "progress";
    for (const auto& pair : sync_pairs){
        oss << "_" << std::get<0>(pair) << "-" << std::get<1>(pair);
    }
    oss << ".npy";
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "saving as " << fname << std::endl;
    EigenMt2npy(calced_laminar, fname);

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

void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname){
    Eigen::MatrixXcd transposed = Mat.transpose();
    // map to const mats in memory
    Eigen::Map<const Eigen::MatrixXcd> MOut(&transposed(0,0), transposed.cols(), transposed.rows());
    // save to np-arrays files
    cnpy::npy_save(fname, MOut.data(), {(size_t)transposed.cols(), (size_t)transposed.rows()}, "w");
}

double shift(double pre_theta, double theta, double rotation_number){
    //forward
    if ((theta - pre_theta) < -M_PI){
        rotation_number += 1;
    }
    //backward
    else if ((theta - pre_theta) > M_PI){
        rotation_number -= 1;
    }

    return rotation_number;
}

bool isLaminar(Eigen::VectorXd phases, std::vector<std::tuple<int, int, double>> sync_pairs){
    bool allSync = true; // flag 
    for (const auto& pair : sync_pairs){
        // if any pair is not sync, allSync is false
        if(std::abs(phases(std::get<0>(pair)-1) - phases(std::get<1>(pair)-1)) > std::get<2>(pair)){
            allSync = false;
            break;
        }
    }
    return allSync;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXcd> calc_next(ShellModel& SM, Eigen::VectorXd pre_n, Eigen::VectorXd pre_theta, Eigen::VectorXcd previous){
    Eigen::VectorXcd now = SM.rk4_(previous);
    Eigen::VectorXd theta = now.cwiseArg();
    Eigen::VectorXd n = pre_n;
    for(int i; i < theta.size(); i++){
        n(i) = shift(pre_theta(i), theta(i), pre_n(i));
    }
    return std::make_tuple(n, theta, now);
}
