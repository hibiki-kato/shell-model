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
#include "Runge_Kutta.hpp"
#include <chrono>
#include <random>
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

void EigenMt2npy(Eigen::MatrixXcd Mat, std::string fname);
Eigen::VectorXcd npy2EigenVec(const char* fname);
Eigen::MatrixXcd npy2EigenMat(const char* fname);

int main(){
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double nu = 0.00018;
    double beta = 0.417;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 1e+4;
    double latter = 1;
    int refresh = 500; // 1000だとカクカク
    int plotDim[] = {4, 5};
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.417_nu0.00018_5000period_dt0.01_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy");
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
            //plot trajectory
            std::map<std::string, std::string> keywords1;
            keywords1.insert(std::make_pair("c", "b")); 
            keywords1.insert(std::make_pair("lw", "0.1"));
            plt::plot(xs[j],ys[j], keywords1);
            //plot clock hand
            plt::xlabel("Real($U_{" + std::to_string(j+1) + "}$)");
            plt::ylabel("Imag($U_{" + std::to_string(j+1) + "}$)");
            std::map<std::string, double> keywords;
            keywords.insert(std::make_pair("hspace", 0.5)); // also right, top, bottom
            keywords.insert(std::make_pair("wspace", 0.5)); // also hspace
            plt::subplots_adjust(keywords);
            std::ostringstream oss;
            oss << "../../animation_frames/real-imag_beta_" << beta << "nu_" << nu << "_" << std::setfill('0') << std::setw(6) << counter << ".png";
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
    oss << "../../animation/real-imag_beta_" << beta << "nu_" << nu <<"_"<< (t-t_0)/latter << "period.mp4";  // 文字列を結合する
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

Eigen::MatrixXcd npy2EigenMat(const char* fname){
    std::string fname_str(fname);
    cnpy::NpyArray arr = cnpy::npy_load(fname_str);
    if (arr.word_size != sizeof(std::complex<double>)) {
        throw std::runtime_error("Unsupported data type in the npy file.");
    }
    Eigen::Map<const Eigen::MatrixXcd> MatT(arr.data<std::complex<double>>(), arr.shape[1], arr.shape[0]);
    return MatT.transpose();
}