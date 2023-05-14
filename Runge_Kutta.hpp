#pragma once
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <math.h>
#include <random>

class ShellModel
{   //data members
    double nu;
    double beta;
    std::complex<double> f;
    double ddt;
    double t_0;
    double t;
    double latter;
    int steps;
    double t_latter_begin;
    Eigen::VectorXd k_n;
    Eigen::VectorXd c_n_1;
    Eigen::VectorXd c_n_2;
    Eigen::VectorXd c_n_3;
    Eigen::VectorXcd x_0;
public:
    //constructor
    ShellModel(double input_nu, double input_beta, std::complex<double> input_f, double input_ddt, double input_t_0, double input_t, double input_latter, Eigen::VectorXcd input_x_0);

    //destructor
    ~ShellModel();
    Eigen::MatrixXcd get_trajectory_();
    // Eigen::VectorXd get_timeline_();

    Eigen::MatrixXcd stagger_and_step();
    

    void set_nu_(double input_nu);
    void set_beta_(double input_beta);
    void set_x_0_(Eigen::VectorXcd input_x_0);
private:
    Eigen::VectorXcd rk4_(Eigen::VectorXcd present);
    Eigen::VectorXcd goy_shell_model_(Eigen::VectorXcd state);
    Eigen::VectorXcd perturbator_(Eigen::VectorXcd state);
};

class Long laminar : public ShellModel
{
public:
    
private:
    bool isLaminar_(Eigen::VectorXcd state, Eigen::MatrixXcd laminar, double epsilon);
}