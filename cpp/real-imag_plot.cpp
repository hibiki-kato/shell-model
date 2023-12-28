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
 * @file real-imag_plot.cpp
 * @author Hibiki Kato
 * @brief Plot each shell on complex plane and save as mp4 animation
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
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include <chrono>
#include <random>
#include "cnpy/cnpy.h"
#include "shared/matplotlibcpp.h"
namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 0.00018;
    params.beta = 0.417;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt =.01;
    double t_0 = 0;
    double t = 1e+4;
    double dump = 1;
    int refresh = 500; // 1000だとカクカク
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../../initials/beta0.417_nu0.00018_5000period_dt0.01_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy");
    ShellModel solver(nu, beta, f, ddt, t_0, t, latter, x_0);
    Eigen::MatrixXcd trajectory = solver.get_trajectory_(); 
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
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(900, 1500);
    // Add graph title
    std::vector<std::vector<double>> xs(trajectory.rows() - 1, std::vector<double>()), ys(trajectory.rows() - 1, std::vector<double>());
    int counter = 0;
    for(int i=0; i < trajectory.cols(); i++){
        if (omp_get_thread_num() == 0) {
            std::cout << "\r processing..." << i << "/" << trajectory.cols() << std::flush;
        }
        for (int j=0; j < trajectory.rows() - 1; j++){
            xs[j].push_back(trajectory(j, i).real());
            ys[j].push_back(trajectory(j, i).imag());
        }
        
        if (i%refresh == 0 && i > trajectory.cols()/10){
            plt::clf();
            for (int j=0; j < trajectory.rows() - 1; j++){
                plt::subplot(5, 3, j+1);
                //plot trajectory
                std::map<std::string, std::string> keywords1;
                keywords1.insert(std::make_pair("c", "b")); 
                keywords1.insert(std::make_pair("lw", "0.1"));
                // keywords1.insert(std::make_pair("alpha", "1"));
                plt::plot(xs[j],ys[j], keywords1);
                //plot clock hand
                std::vector<double> x = {0, xs[j].back()};
                std::vector<double> y = {0, ys[j].back()};
                std::map<std::string, std::string> keywords2;
                keywords2.insert(std::make_pair("c", "r")); 
                keywords2.insert(std::make_pair("lw", "3.0"));
                plt::plot(x, y, keywords2);
                plt::xlabel("Real($U_{" + std::to_string(j+1) + "}$)");
                plt::ylabel("Imag($U_{" + std::to_string(j+1) + "}$)");
            }
            std::map<std::string, double> keywords;
            keywords.insert(std::make_pair("hspace", 0.5)); // also right, top, bottom
            keywords.insert(std::make_pair("wspace", 0.5)); // also hspace
            plt::subplots_adjust(keywords);
            std::ostringstream oss;
            oss << "../../animation_frames/real-imag_beta" << params.beta << "nu" << params.nu << "_" << std::setfill('0') << std::setw(6) << counter << ".png";
            counter++;
            std::string plotfname = oss.str(); // 文字列を取得する
            std::cout << "Saving result to " << plotfname << std::endl;
            plt::save(plotfname);
            oss.str("");
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
    oss << "../../animation/real-imag_beta" << params.beta << "nu" << params.nu <<"_"<< t-t_0 << "period.mp4";  // 文字列を結合する
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

    return 0;


    myfunc::duration(start);
}