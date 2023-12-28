/**
 * @file animation.cpp
 * @author Hibiki Kato
 * @brief Plot animation of the trajectory
 * @version 0.1
 * @date 2023-09-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <iomanip> // include this header for std::setw() and std::setfill()
#include <fstream>
#include <sstream>
#include <filesystem>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <chrono>
#include <random>
#include "cnpy/cnpy.h"
#include "shared/Flow.hpp"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/myFunc.hpp"
namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 0.00018;
    params.beta = 0.416;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+4;
    double dump = 1e+3;
    int refresh =  2000;
    int plotDim[] = {4, 5};
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../../initials/beta0.417_nu0.00018_5000period_dt0.01_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy");

    ShellModel SM(params, dt, t_0, t, dump, x_0);
    Eigen::MatrixXcd trajectory = SM.get_trajectory(); 
    std::cout << "trajectory size: " << trajectory.rows() << "x" << trajectory.cols() << std::endl;
    // Eigen::MatrixXcd trajectory = npy2EigenMat("../../generated_lam/generated_laminar_beta_0.419nu_0.00018_200000period1500check500progresseps0.1.npy");
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
    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    // vector for plotting
    std::vector<double> x, y;
    int counter = 0;
    for(int i=0; i < trajectory.cols(); i++){
        std::cout << "\r processing..." << i << "/" << trajectory.cols() << std::flush;
        x.push_back(std::abs(trajectory(plotDim[0]-1, i)));
        y.push_back(std::abs(trajectory(plotDim[1]-1, i)));
        if (i%refresh == 0){
            //plot trajectory
            std::map<std::string, std::string> keywords;
            keywords.insert(std::make_pair("c", "b")); 
            keywords.insert(std::make_pair("lw", "1"));
            plt::figure_size(1000, 1000);
            plt::plot(x, y, keywords);
            //plot clock hand
            plt::xlabel("$|U_{" + std::to_string(plotDim[0]) + "}|$");
            plt::ylabel("$|U_{" + std::to_string(plotDim[1]) + "}|$");
            std::ostringstream oss;
            oss << "../../animation_frames/beta_" << params.beta << "nu_" << params.nu << "_" << std::setfill('0') << std::setw(6) << counter << ".png";
            counter++;
            std::string plotfname = oss.str(); // 文字列を取得する
            // std::cout << "Saving result to " << plotfname << std::endl;
            plt::save(plotfname);
            oss.str("");
            plt::close();
        }
    }

   /*
      ████                                     █       █               ██      ██  █████      ██
     ██                                        █       █               ███    ███  ██  ██    ███
    ██       ████   █ ███  █    █   ███   █ ██████    ████   ████      ███    ███  ██   █   ████
    █       ██  ██  ██  ██  █   █  ██  █  ██   █       █    ██  ██     █ █   ████  ██  ██   █ ██
    █       █    █  █    █  █  ██  █   ██ █    █       █    █    █     █  █  █ ██  █████   █  ██
    █       █    █  █    █  ██ █   ██████ █    █       █    █    █     █  █  █ ██  ██     ██  ██
    ██      █    █  █    █   █ █   █      █    █       █    █    █     █  ███  ██  ██     ███████
     ██     ██  ██  █    █   ███   ██     █    ██      ██   ██  ██     █   ██  ██  ██         ██
      ████   ████   █    █   ██     ████  █     ██      ██   ████      █   ██  ██  ██         ██
   */
    std::vector<std::string> imagePaths;
    std::ostringstream oss;
    oss << "../../animation/beta_" << params.beta << "nu_" << params.nu <<"_"<< t-t_0 << "_" << plotDim[0] << "-" << plotDim[1] << "period.mp4";  // 文字列を結合する
    std::string outputFilename =  oss.str();
    std::string folderPath = "../../animation_frames/"; // フォルダのパスを指定
    int framerate = 60; // フレーム間の遅延時間（ミリ秒）

    // フォルダ内のファイルを取得
    
    std::string command = "ffmpeg -framerate " + std::to_string(framerate) + " -pattern_type glob -i '" + folderPath + "*.png' -c:v libx264 -pix_fmt yuv420p " + outputFilename;
    std::cout << command << std::endl;
    // コマンドを実行
    int result = std::system(command.c_str());
    if (result == 0) {
        std::cout << "MP4 animation created successfully." << std::endl;
        // フォルダ内のファイルを削除
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            fs::remove(entry.path());
        }

        std::cout << "All files in the animation_frames/ have been deleted." << std::endl;
    } else {
        std::cerr << "Failed to create Mp4 animation." << std::endl;
    }
    myfunc::duration(start);
}