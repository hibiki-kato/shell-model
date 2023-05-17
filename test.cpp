#include <eigen3/Eigen/Dense>
#include <complex>
#include <iostream>
#include <vector>
#include <random>
#include <type_traits>
// #include <cmath>
#include "matplotlibcpp.h"
#include "cnpy/cnpy.h"
namespace plt = matplotlibcpp;

Eigen::VectorXcd perturbator(Eigen::VectorXcd state);

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
    ShellModel(double input_nu, double input_beta, std::complex<double> input_f, double input_ddt, double input_t_0, double input_t, double input_latter, Eigen::VectorXcd input_x_0){
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
    ~ShellModel(){
    }
    

    Eigen::MatrixXcd get_trajectory_(){
        int row = x_0.rows() + 1;
        Eigen::MatrixXcd trajectory(row, steps+1);
        double time = t_0;

        //set initial point
        trajectory.block(0, 0, row-1, 1) = x_0;
        trajectory(row-1, 0) = time;
        //renew x_0 while reaching latter
        for (int i = 0; i < static_cast<int>((t - t_0) / ddt +0.5) - steps; i++){
            trajectory.block(0, 0, row-1, 1) = rk4_(trajectory.block(0, 0, row-1, 1));
            trajectory(row-1, 0) = time;
            time += ddt;
        }
        
        //solve
        for(int i = 0; i < steps; i++){
            trajectory.block(0, i+1, row-1, 1) = rk4_(trajectory.block(0, i, row-1, 1));
            trajectory(row-1, i+1) = time;
            time += ddt;
        }
        return trajectory;
    };

    void set_nu_(double input_nu){
        nu = input_nu;
    }
    void set_beta_(double input_beta){
        beta = input_beta;
        int dim = x_0.rows();
        // update c_n_2 and c_n_3
        c_n_2 = Eigen::VectorXd::Zero(dim);
        c_n_2.middleRows(1, dim-2) = k_n.middleRows(2, dim-2) * (-beta);

        c_n_3 = Eigen::VectorXd::Zero(dim);
        c_n_3.bottomRows(dim-2) = k_n.middleRows(2, dim-2) * (beta - 1);
    }

    void set_x_0_(Eigen::VectorXcd input_x_0){
        x_0 = input_x_0;
    }
private:
   Eigen::VectorXcd rk4_(Eigen::VectorXcd present)
    {
        Eigen::VectorXcd k1 = ddt * goy_shell_model_(present);
        Eigen::VectorXcd k2 = ddt * goy_shell_model_(present.array() + k1.array() /2);
        Eigen::VectorXcd k3 = ddt * goy_shell_model_(present.array() + k2.array() /2);
        Eigen::VectorXcd k4 = ddt * goy_shell_model_(present.array() + k3.array());
        return present.array() + (k1.array() + 2 * k2.array() + 2 * k3.array() + k4.array()) / 6;
    }

    Eigen::VectorXcd goy_shell_model_(Eigen::VectorXcd state)
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
};


int main(){
    double nu = 0.0001732;
    double beta = 0.417;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 100000;
    double latter = 1;
    Eigen::VectorXcd x_0(14);
    x_0(0) = std::complex<double>(0.4350E+00 , 0.5008E+00);
    x_0(1) = std::complex<double>(0.1259E+00 , 0.2437E+00);
    x_0(2) = std::complex<double>(-0.8312E-01 , -0.4802E-01);
    x_0(3) = std::complex<double>(0.5164E-01 , -0.1599E+00);
    x_0(4) = std::complex<double>(-0.1899E+00 , -0.3602E-01);
    x_0(5) = std::complex<double>(0.4093E-03 , 0.8506E-01);
    x_0(6) = std::complex<double>(0.9539E-01 , 0.3215E-01);
    x_0(7) = std::complex<double>(-0.5834E-01 , 0.4433E-01);
    x_0(8) = std::complex<double>(-0.8790E-02 , 0.2502E-01);
    x_0(9) = std::complex<double>(0.3385E-02 , 0.1148E-02);
    x_0(10) = std::complex<double>(-0.7072E-04 , 0.5598E-04);
    x_0(11) = std::complex<double>(-0.5238E-07 , 0.1467E-06);
    x_0(12) = std::complex<double>(0.1E-07 ,0.1E-06);
    x_0(13) = std::complex<double>(0.1E-07 ,0.1E-06);

    
    ShellModel solver(nu, beta, f, ddt, t_0, t, latter, x_0);

    // load npz
    cnpy::NpyArray loaded = cnpy::npy_load("beta_0.417nu_0.00017256_100000period.npy");
    Eigen::Map<const Eigen::MatrixXcd> Loaded(loaded.data<std::complex<double>>(), loaded.shape[0], loaded.shape[1]);
    Eigen::MatrixXcd trajectory = Loaded;

    // Eigen::MatrixXcd trajectory = solver.get_trajectory_();
    
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1200, 780);
    // Add graph title
    plt::title("Sample figure");
    std::vector<double> x(trajectory.cols()),y(trajectory.cols());

    for(int i=0;i<trajectory.cols();i++){
        x[i]=trajectory.cwiseAbs()(14, i);
        y[i]=trajectory.cwiseAbs()(0, i);
    }

    // plt::plot(x,y);
    Eigen::VectorXd state;
    std::vector<double> u_0(10000);
    std::vector<double> u_1(10000);
    for(int i = 0; i < 10000; i++){
        state = perturbator(x_0).cwiseAbs();
        u_0[i] = state(0);
        u_1[i] = state(1);
    }
    plt::scatter(u_0, u_1);
    const char* filename = "test.png";
    std::cout << "Saving result to " << filename << std::endl;
    plt::save(filename);


    
}


Eigen::VectorXcd perturbator(Eigen::VectorXcd state){   
    std::random_device rd;
    std::mt19937 gen(rd());
    double a = -3;
    double b = -10;
    std::uniform_real_distribution<double> s(-1, 1);
    std::uniform_real_distribution<double> dis(b, a);

    Eigen::VectorXd unit = Eigen::VectorXd::Ones(state.rows());
    for(int i = 0; i < state.rows(); i++){
        unit(i) = s(gen);
    }

    Eigen::VectorXcd u = state.cwiseProduct(unit);
    u /= u.norm();

    return (u.array() * std::pow(10, dis(gen)) + state.array()).matrix();

};
