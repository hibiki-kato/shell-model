#include "Runge_Kutta.hpp"
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <math.h>
#include <random>

//constructor
CoupledRossler::CoupledRossler(double input_nu, double input_beta, std::complex<double> input_f, double input_ddt, double input_t_0, double input_t, double input_latter, Eigen::VectorXcd input_x_0){
    ddt = input_ddt;
    t_0 = input_t_0;
    t = input_t;
    latter = input_latter;
    
    // make k_n and c_n using beta
    int dim = x_0.rows();
    
    steps = static_cast<long>((t - t_0) / ddt / latter+ 0.5);
    t_latter_begin = t - (t - t_0) / latter;
 }
//destructor
CoupledRossler::~CoupledRossler(){
 }

Eigen::MatrixXcd CoupledRossler::get_trajectory_(){
    int row = x_0.rows() + 1;
    Eigen::MatrixXcd trajectory(row, steps+1);
    double time = t_0;

    //set initial point
    trajectory.block(0, 0, row-1, 1) = x_0;
    trajectory(row-1, 0) = time;
    //renew x_0 while reaching latter
    for (long i = 0; i < static_cast<long>((t - t_0) / ddt +0.5) - steps; i++){
        trajectory.block(0, 0, row-1, 1) = CoupledRossler::rk4_(trajectory.block(0, 0, row-1, 1));
        trajectory(row-1, 0) = time;
        time += ddt;
    }

    //solve

    for(long i = 0; i < steps; i++){
        trajectory.block(0, i+1, row-1, 1) = CoupledRossler::rk4_(trajectory.block(0, i, row-1, 1));
        trajectory(row-1, i+1) = time;
        time += ddt;
    }
    return trajectory;
};

Eigen::VectorXd CoupledRossler::get_energy_spectrum_(const Eigen::MatrixXcd& trajectory){
    if (trajectory.rows() == 0){
        Eigen::VectorXcd x = x_0;
        Eigen::VectorXd sum(x_0.rows(), 1);
        
        //renew x_0 while reaching latter
        for (long i = 0; i < static_cast<long>((t - t_0) / ddt +0.5) - steps; i++){
            x = CoupledRossler::rk4_(x);
        }

        // get energy spectrum by calc mean of absolute value of each shell's
        sum = x.cwiseAbs();
        for (long i = 0; i < steps; i++){
            x = CoupledRossler::rk4_(x);
            sum += x.cwiseAbs();
        }
        return sum.array() / (steps+1);
    }
    else{
        Eigen::VectorXd sum(trajectory.rows()-1, 1);
        sum = trajectory.topRows(trajectory.rows()-1).cwiseAbs().rowwise().mean();
        return sum;
    }
    
}

void CoupledRossler::set_nu_(double input_nu){
    nu = input_nu;
}
void CoupledRossler::set_beta_(double input_beta){
    beta = input_beta;
    int dim = x_0.rows();
    // update c_n_2 and c_n_3
    c_n_2 = Eigen::VectorXd::Zero(dim);
    c_n_2.middleRows(1, dim-2) = k_n.topRows(dim-2).array() * (-beta);

    c_n_3 = Eigen::VectorXd::Zero(dim);
    c_n_3.bottomRows(dim-2) = k_n.topRows(dim-2).array() * (beta - 1);
}
void CoupledRossler::set_t_0_(double input_t_0){
    t_0 = input_t_0;
    steps = static_cast<long>((t - t_0) / ddt / latter+ 0.5);
    t_latter_begin = t - (t - t_0) / latter;
}
void CoupledRossler::set_t_(double input_t){
    t = input_t;
    steps = static_cast<long>((t - t_0) / ddt / latter+ 0.5);
    t_latter_begin = t - (t - t_0) / latter;
}
void CoupledRossler::set_steps_(long input_steps){
    steps = input_steps;
}
void CoupledRossler::set_x_0_(Eigen::VectorXcd input_x_0){
    x_0 = input_x_0;
}
double CoupledRossler::get_beta_(){
    return beta;
}
double CoupledRossler::get_nu_(){
    return nu;
}

double CoupledRossler::get_ddt_(){
    return ddt;
}
double CoupledRossler::get_t_0_(){
    return t_0;
}
double CoupledRossler::get_t_(){
    return t;
}
long CoupledRossler::get_steps_(){
    return steps;
}
Eigen::VectorXd CoupledRossler::get_k_n_(){
    return k_n;
}

Eigen::VectorXcd CoupledRossler::get_x_0_(){
    return x_0;
}


Eigen::VectorXcd CoupledRossler::rk4_(Eigen::VectorXcd present){
    Eigen::VectorXcd k1 = ddt * CoupledRossler::goy_shell_model_(present).array();
    Eigen::VectorXcd k2 = ddt * CoupledRossler::goy_shell_model_(present.array() + k1.array() /2).array();
    Eigen::VectorXcd k3 = ddt * CoupledRossler::goy_shell_model_(present.array() + k2.array() /2).array();
    Eigen::VectorXcd k4 = ddt * CoupledRossler::goy_shell_model_(present.array() + k3.array()).array();
    return present.array() + (k1.array() + 2 * k2.array() + 2 * k3.array() + k4.array()) / 6;
}

Eigen::VectorXcd CoupledRossler::goy_shell_model_(Eigen::VectorXcd state){
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