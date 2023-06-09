#pragma once
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <math.h>
#include <iostream>
#include <regex>
#include <random>
#include <omp.h>

class ShellModel{   
public:
    //constructor
    ShellModel(double input_nu, double input_beta, std::complex<double> input_f, double input_ddt, double input_t_0, double input_t, double input_latter, Eigen::VectorXcd input_x_0);

    //destructor
    ~ShellModel();
    Eigen::MatrixXcd get_trajectory_();
    Eigen::VectorXd get_energy_spectrum_();
    Eigen::VectorXcd rk4_(Eigen::VectorXcd present);

    void set_nu_(double input_nu);
    void set_beta_(double input_beta);
    void set_t_0_(double input_t_0);
    void set_t_(double input_t);
    void set_steps_(long input_steps);
    void set_x_0_(Eigen::VectorXcd input_x_0);

    double get_beta_();
    double get_nu_();
    double get_ddt_();
    double get_t_0_();
    double get_t_();
    long get_steps_();
    Eigen::VectorXd get_k_n_();
    Eigen::VectorXcd get_x_0_();
private:
    //data members
    double nu;
    double beta;
    std::complex<double> f;
    double ddt;
    double t_0;
    double t;
    double latter;
    long steps;
    double t_latter_begin;
    Eigen::VectorXd k_n;
    Eigen::VectorXd c_n_1;
    Eigen::VectorXd c_n_2;
    Eigen::VectorXd c_n_3;
    Eigen::VectorXcd x_0;

    Eigen::VectorXcd goy_shell_model_(Eigen::VectorXcd state);
};




class LongLaminar : public ShellModel{
public:
    LongLaminar(double input_nu, double input_beta, std::complex<double> input_f, double input_ddt, double input_t_0, double input_t, double input_latter, Eigen::VectorXcd input_x_0, Eigen::MatrixXcd input_laminar, double input_epsilon, int input_skip, double input_check_sec, double input_progress_sec, int input_threads);
    ~LongLaminar();
    Eigen::MatrixXcd stagger_and_step_();
    bool isLaminarTrajectory_(Eigen::MatrixXcd trajectory);
    std::vector<double> laminar_duration_(const Eigen::MatrixXcd& trajectory = Eigen::MatrixXcd());
    std::vector<double> laminar_duration_logged_(const Eigen::MatrixXcd& trajectory = Eigen::MatrixXcd());
    double laminar_persistent_(Eigen::MatrixXcd trajectory);
    Eigen::VectorXcd perturbator_(Eigen::VectorXcd state, int s_min = -3, int s_max = -10);
    bool isLaminarPoint_(Eigen::VectorXcd state);
    Eigen::MatrixXcd extractor(const Eigen::MatrixXcd& trajectory, int index, double back, double forward);

private:
    Eigen::MatrixXcd laminar;
    double epsilon;
    int skip;
    double check_sec;
    double progress_sec;
    double begin_time_of_stag_and_step;
    double end_time_of_stag_and_step;
    int threads;
    void EigenVecXcd2npy(Eigen::VectorXcd Vec, std::string fname);


    
};