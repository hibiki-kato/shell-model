#include "Runge_Kutta.hpp"
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <math.h>
#include <random>

//constructor
ShellModel::ShellModel(double input_nu, double input_beta, std::complex<double> input_f, double input_ddt, double input_t_0, double input_t, double input_latter, Eigen::VectorXcd input_x_0){
    nu = input_nu;
    beta = input_beta;
    f = input_f;
    ddt = input_ddt;
    t_0 = input_t_0;
    t = input_t;
    latter = input_latter;
    x_0 = input_x_0;
    
    // make k_n and c_n using beta
    int dim = x_0.rows();
    k_n = Eigen::VectorXd::Zero(dim);
    double q = 2.0;
    double k_0 = pow(2, -4);
    
    for (int i = 0; i < dim; i++) {
        k_n(i) = k_0 * pow(q, i+1);
    };
    c_n_1 = Eigen::VectorXd::Zero(dim);
    c_n_1.topRows(dim-2) = k_n.topRows(dim-2);

    c_n_2 = Eigen::VectorXd::Zero(dim);
    c_n_2.middleRows(1, dim-2) = k_n.topRows(dim-2).array() * (-beta);

    c_n_3 = Eigen::VectorXd::Zero(dim);
    c_n_3.bottomRows(dim-2) = k_n.topRows(dim-2).array() * (beta - 1);
    steps = static_cast<int>((t - t_0) / ddt / latter+ 0.5);
    t_latter_begin = t - (t - t_0) / latter;
 }
//destructor
ShellModel::~ShellModel(){
 }

Eigen::MatrixXcd ShellModel::get_trajectory_(){
    int row = x_0.rows() + 1;
    Eigen::MatrixXcd trajectory(row, steps+1);
    double time = t_0;

    //set initial point
    trajectory.block(0, 0, row-1, 1) = x_0;
    trajectory(row-1, 0) = time;
    //renew x_0 while reaching latter
    for (int i = 0; i < static_cast<int>((t - t_0) / ddt +0.5) - steps; i++){
        trajectory.block(0, 0, row-1, 1) = ShellModel::rk4_(trajectory.block(0, 0, row-1, 1));
        trajectory(row-1, 0) = time;
        time += ddt;
    }

    //solve

    for(int i = 0; i < steps; i++){
        trajectory.block(0, i+1, row-1, 1) = ShellModel::rk4_(trajectory.block(0, i, row-1, 1));
        trajectory(row-1, i+1) = time;
        time += ddt;
    }
    return trajectory;
};

Eigen::VectorXd ShellModel::get_energy_spectrum_(){
    Eigen::VectorXcd x = x_0;
    Eigen::VectorXd sum(x_0.rows(), 1);
    
    //renew x_0 while reaching latter
    for (int i = 0; i < static_cast<int>((t - t_0) / ddt +0.5) - steps; i++){
        x = ShellModel::rk4_(x);
    }

    // get energy spectrum by calc mean of absolute value of each shell's
    sum = x.cwiseAbs();
    for (int i = 0; i < steps; i++){
        x = ShellModel::rk4_(x);
        sum += x.cwiseAbs();
    }
    return sum.array() / (steps+1);
}

void ShellModel::set_nu_(double input_nu){
    nu = input_nu;
}
void ShellModel::set_beta_(double input_beta){
    beta = input_beta;
    int dim = x_0.rows();
    // update c_n_2 and c_n_3
    c_n_2 = Eigen::VectorXd::Zero(dim);
    c_n_2.middleRows(1, dim-2) = k_n.middleRows(2, dim-2) * (-beta);

    c_n_3 = Eigen::VectorXd::Zero(dim);
    c_n_3.bottomRows(dim-2) = k_n.middleRows(2, dim-2) * (beta - 1);
}

void ShellModel::set_x_0_(Eigen::VectorXcd input_x_0)
{
    x_0 = input_x_0;
}

Eigen::VectorXd ShellModel::get_k_n_()
{
    return k_n;
}


Eigen::VectorXcd ShellModel::rk4_(Eigen::VectorXcd present)
{
    Eigen::VectorXcd k1 = ddt * ShellModel::goy_shell_model_(present).array();
    Eigen::VectorXcd k2 = ddt * ShellModel::goy_shell_model_(present.array() + k1.array() /2).array();
    Eigen::VectorXcd k3 = ddt * ShellModel::goy_shell_model_(present.array() + k2.array() /2).array();
    Eigen::VectorXcd k4 = ddt * ShellModel::goy_shell_model_(present.array() + k3.array()).array();
    return present.array() + (k1.array() + 2 * k2.array() + 2 * k3.array() + k4.array()) / 6;
}

Eigen::VectorXcd ShellModel::goy_shell_model_(Eigen::VectorXcd state)
{
    int dim = state.rows();
    Eigen::VectorXcd u = Eigen::VectorXd::Zero(dim+4);

    u.middleRows(2, state.rows()) = state;
    Eigen::VectorXcd ddt_u = (c_n_1.array() * u.middleRows(3,dim).conjugate().array() * u.bottomRows(dim).conjugate().array()
                            + c_n_2.array() * u.middleRows(1,dim).conjugate().array() * u.middleRows(3,dim).conjugate().array()
                            + c_n_3.array() * u.middleRows(1,dim).conjugate().array() * u.topRows(dim).conjugate().array()) * std::complex<double>(0, 1.0)
                            - nu * u.middleRows(2,dim).array() * k_n.array().square();
    ddt_u(0) += f;
    return ddt_u;
}
