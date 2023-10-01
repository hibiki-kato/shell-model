#include "Runge_Kutta.hpp"
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <math.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <random>
#include <omp.h>
#include "cnpy/cnpy.h"

LongLaminar::LongLaminar(double input_nu, double input_beta, std::complex<double> input_f, double input_ddt, double input_t_0, double input_t, double input_latter, Eigen::VectorXcd input_x_0, Eigen::MatrixXcd input_laminar, double input_epsilon, int input_skip, double input_check_sec, double input_progress_sec, int input_threads = omp_get_max_threads()) : ShellModel(input_nu, input_beta, input_f, input_ddt, input_t_0, input_t, input_latter, input_x_0){
    laminar = input_laminar;
    epsilon = input_epsilon;
    skip = input_skip;
    check_sec = input_check_sec;
    progress_sec = input_progress_sec;
    begin_time_of_stag_and_step = input_t_0;
    end_time_of_stag_and_step = input_t;
    threads = input_threads;
}

LongLaminar::~LongLaminar(){
}

Eigen::MatrixXcd LongLaminar::stagger_and_step_(){
    Eigen::MatrixXcd staggered_traj = Eigen::MatrixXcd::Zero(ShellModel::get_x_0_().size()+1, ShellModel::get_steps_()+1); // trajectory going to be generated by stagger and step
    int stagger_and_step_num = static_cast<int>((end_time_of_stag_and_step - begin_time_of_stag_and_step) / progress_sec + 0.5); // times of stagger and step
    int check_steps = static_cast<int>(check_sec / ShellModel::get_ddt_() + 0.5); //steps of checked trajectory
    int progress_steps = static_cast<int>(progress_sec / ShellModel::get_ddt_() + 0.5); //steps of progress
    int cycle_limit = 5E+04;
    // ShellModel::set_steps_(check_steps); //　多分いらない?
    
    for (int i = 0; i < stagger_and_step_num; i++){
        std::cout << "\r 現在" << ShellModel::get_t_0_() << "時間" << std::flush;
        ShellModel::set_t_(ShellModel::get_t_0_() + check_sec);

        // no perturbation at first
        Eigen::MatrixXcd checked_traj = ShellModel::get_trajectory_();
        if (LongLaminar::isLaminarTrajectory_(checked_traj)){
            staggered_traj.middleCols(i*progress_steps, progress_steps+1) = checked_traj.leftCols(progress_steps+1);
            ShellModel::set_t_0_(ShellModel::get_t_0_() + progress_sec);
            ShellModel::set_x_0_(checked_traj.block(0, progress_steps, ShellModel::get_x_0_().size(), 1));
        }
        else{
            std::cout << std::endl;
            bool exit = false;
            // parallel
            #pragma omp parallel num_threads(threads)
            {   int local_threads = threads;
                int local_counter = 0;
                int local_cycle_limit = cycle_limit;
                int thread_id = omp_get_thread_num();
                while (true) {
                    local_counter++;
                    ShellModel::set_x_0_(LongLaminar::perturbation_(ShellModel::get_x_0_()));
                    Eigen::MatrixXcd checked_traj = ShellModel::get_trajectory_();
                    if (LongLaminar::isLaminarTrajectory_(checked_traj)) {
                        #pragma omp single 
                        {
                            staggered_traj.middleCols(i * progress_steps, progress_steps + 1) = checked_traj.leftCols(progress_steps + 1);
                            ShellModel::set_t_0_(ShellModel::get_t_0_() + progress_sec);
                            ShellModel::set_x_0_(checked_traj.block(0, progress_steps, ShellModel::get_x_0_().size(), 1));
                            std::cout << std::endl;
                        }
                        break;
                    }
                    if (local_counter >= local_cycle_limit / local_threads) {
                        #pragma omp single
                        exit = true;
                        break;
                    }
                    if (thread_id == 0){
                        std::cout << "\r 現在試行" << local_counter * local_threads << std::flush;
                    }
                }
            }   
            if (exit){
                return staggered_traj.leftCols(i*progress_steps+1);
            }
        }
    }
    return staggered_traj;
}

bool LongLaminar::isLaminarPoint_(Eigen::VectorXcd state){
// if the points is in laminar flow, return true. Otherwise return false.  state can include time bcause it will be dropped.
    int row_start = 0;
    int row_end = 9;
    Eigen::VectorXd distance = (laminar.middleRows(row_start, row_end).cwiseAbs() - state.middleRows(row_start, row_end).replicate(1, laminar.cols()).cwiseAbs()).colwise().norm();

    


    return (distance.array() < epsilon).any();
}

bool LongLaminar::isLaminarTrajectory_(Eigen::MatrixXcd trajectory){
    for (int i = 0; i < trajectory.cols()/skip; i++){
        if (!isLaminarPoint_(trajectory.col(i*skip))) {
            return false;
        }
    }
    return true;
}
Eigen::VectorXcd LongLaminar::perturbation_(Eigen::VectorXcd state, int s_min, int s_max){   
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> s(-1, 1);
    std::uniform_real_distribution<double> dis(s_min, s_max);

    Eigen::VectorXd unit = Eigen::VectorXd::Ones(state.rows());
    for(int i = 0; i < state.rows(); i++){
        unit(i) = s(gen);
    }

    Eigen::VectorXcd u = state.cwiseProduct(unit);
    u /= u.norm();

    return (u.array() * std::pow(10, dis(gen)) + state.array()).matrix();

}

std::vector<double> LongLaminar::laminar_duration_(const Eigen::MatrixXcd& trajectory){
    int counter = -1;
    std::vector<double> durations;
    // 軌道が与えられていない場合
    if (trajectory.rows() == 0){

        Eigen::VectorXcd x = LongLaminar::get_x_0_();
        for (long i = 0; i < LongLaminar::get_steps_(); i++) {
            x = LongLaminar::rk4_(x);
            
            if (i % skip == 0) {
                if (LongLaminar::isLaminarPoint_(x) == 1) {
                    counter++;
                } else {
                    // カウンターが1以上の時のみ記録
                    if (counter > 0) {
                        durations.push_back(counter * LongLaminar::get_ddt_() * skip);
                        counter = -1;
                    }
                    // カウンターをリセット
                    counter = -1;
                }
            }
        }

        // 途切れた場合は最後の点までの時間を記録 (暫定的処理)
        if (counter > 0){
            durations.push_back(counter * LongLaminar::get_ddt_() * skip);
        }
        // 一度もラミナーに入らなかった場合は0を入れる
        if (durations.size() == 0){
            durations.push_back(0);
        }

    }

    // 軌道が与えられている場合
    else{
        int check_times = trajectory.cols()/skip + 1;
        for(long i =0; i < check_times; i++){
            if (LongLaminar::isLaminarPoint_(trajectory.col(i*skip))){
                counter++;
            }
            else{
                if (counter > 0){
                    durations.push_back(counter * LongLaminar::get_ddt_() * skip);
                    counter = -1;
                }
                counter = -1;
            }
        }
        // 途切れた場合は最後の点までの時間を記録 (暫定的処理)
        if (counter > 0){
            durations.push_back(counter * LongLaminar::get_ddt_() * skip);
        }
        // 一度もラミナーに入らなかった場合は0を入れる
        if (durations.size() == 0){
            durations.push_back(0);
        }
    }
    return durations;
}

double LongLaminar::laminar_persistent_(Eigen::MatrixXcd trajectory){
    double duration = 0;
    for (int i = 0; i < trajectory.cols()/skip; i++){
        if (!isLaminarPoint_(trajectory.col(i*skip))) {
            return duration;
        }
        duration += ShellModel::get_ddt_() * skip;
    }
    return duration;
}

Eigen::MatrixXcd LongLaminar::extractor(const Eigen::MatrixXcd& trajectory, int index, double back, double forward){
    int backIndex = index - static_cast<int>(std::round(back / ShellModel::get_ddt_()));
    if (backIndex < 0){
        throw std::runtime_error("Backward time is out of range.");
    }
    int forwardIndex = index + static_cast<int>(std::round(forward / ShellModel::get_ddt_()));
    if (forwardIndex > trajectory.cols()){
        throw std::runtime_error("Forward time is out of range.");
    }
    return trajectory.middleCols(backIndex, forwardIndex - backIndex + 1);
}

std::vector<double> LongLaminar::laminar_duration_logged_(const Eigen::MatrixXcd& trajectory){
    int counter = -1;
    std::vector<double> durations;
    // 軌道が与えられていない場合
    if (trajectory.rows() == 0){

        Eigen::VectorXcd x = LongLaminar::get_x_0_();
        std::ostringstream oss;
        for (long i = 0; i < LongLaminar::get_steps_(); i++) {
            x = LongLaminar::rk4_(x);
            
            if (i % skip == 0) {
                if (LongLaminar::isLaminarPoint_(x) == 1) {
                    counter++;
                    if (counter == 1){
                        oss << "../../initials/beta" << LongLaminar::get_beta_() << "nu" << LongLaminar::get_nu_() << "_" << (counter-1) * LongLaminar::get_ddt_() * skip << "period.npy";
                        LongLaminar::EigenVecXcd2npy(x, oss.str());
                    }
                } else {
                    // カウンターが1以上の時のみ記録
                    if (counter > 0) {
                        durations.push_back(counter * LongLaminar::get_ddt_() * skip);
                        std::string oldName = oss.str();
                        oss.str("");
                        oss << "../../initials/beta" << LongLaminar::get_beta_() << "nu" << LongLaminar::get_nu_() << "_" << (counter-1) * LongLaminar::get_ddt_() * skip << "period.npy";
                        std::string newName = oss.str();
                        int result = std::rename(oldName.c_str(), newName.c_str());

                        if (result != 0) {
                            std::cerr << "ファイル名の変更に失敗しました。" << std::endl;
                        }
                        oss.str("");
                        counter = -1;
                    }
                    // カウンターをリセット
                    counter = -1;
                }
            }
        }

        // 途切れた場合は最後の点までの時間を記録 (暫定的処理)
        if (counter > 0){
            durations.push_back(counter * LongLaminar::get_ddt_() * skip);
        }
        // 一度もラミナーに入らなかった場合は0を入れる
        if (durations.size() == 0){
            durations.push_back(0);
        }
    }
    return durations;
}

void LongLaminar::EigenVecXcd2npy(Eigen::VectorXcd Vec, std::string fname){
    std::vector<std::complex<double>> x(Vec.size());
    for(int i=0;i<Vec.size();i++){
        x[i]=Vec(i);
    }
    cnpy::npy_save(fname, &x[0], {(size_t)Vec.size()}, "w");
}