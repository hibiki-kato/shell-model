              █  █     █                    █         █                        █
           █  █  █     █                    █         █         ██             █
              █  █     █                    █         █         ██             █
██  ██  █  █  █  █     █████   ███       ████   ███   █   ███  ████  ███    ████
 █  ██  █  █  █  █     ██  █  ██  █     ██  █  ██  █  █  ██  █  ██  ██  █  ██  █
 █  ██  █  █  █  █     █   ██ █   █     █   █  █   █  █  █   █  ██  █   █  █   █
 █ █ █ ██  █  █  █     █   ██ █████     █   █  █████  █  █████  ██  █████  █   █
 ███  ██   █  █  █     █   ██ █         █   █  █      █  █      ██  █      █   █
  ██  ██   █  █  █     ██  █  ██  █     ██  █  ██  █  █  ██  █  ██  ██  █  ██  █
  █   ██   █  █  █     █████   ████      ████   ████  █   ████   ██  ████   ████
#pragma once
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <math.h>
#include <iostream>
#include <random>
#include <omp.h>


// class LongLaminar : public ShellModel{
// public:
//     LongLaminar(double input_nu, double input_beta, std::complex<double> input_f, double input_dt, double input_t_0, double input_t, double input_latter, Eigen::VectorXcd input_x_0, Eigen::MatrixXcd input_laminar, double input_epsilon, int input_skip, double input_check_sec, double input_progress_sec, int input_threads);
//     ~LongLaminar();
//     Eigen::MatrixXcd stagger_and_step_();
//     bool isLaminarTrajectory_(Eigen::MatrixXcd trajectory);
//     std::vector<double> laminar_duration_(const Eigen::MatrixXcd& trajectory = Eigen::MatrixXcd());
//     std::vector<double> laminar_duration_logged_(const Eigen::MatrixXcd& trajectory = Eigen::MatrixXcd());
//     double laminar_persistent_(Eigen::MatrixXcd trajectory);
//     Eigen::VectorXcd perturbation_(Eigen::VectorXcd state, int s_min = -10, int s_max = -3);
//     bool isLaminarPoint_(Eigen::VectorXcd state);
//     Eigen::MatrixXcd extractor(const Eigen::MatrixXcd& trajectory, int index, double back, double forward);

// private:
//     Eigen::MatrixXcd laminar;
//     double epsilon;
//     int skip;
//     double check_sec;
//     double progress_sec;
//     double begin_time_of_stag_and_step;
//     double end_time_of_stag_and_step;
//     int threads;
//     void EigenVecXcd2npy(Eigen::VectorXcd Vec, std::string fname);    
// };