/**
 * @file jacobi_plot.cpp
 * @author Hibiki Kato
 * @brief ランダムなヤコビ行列を作りそこから毎ステップ選んで時間発展させる．アニメーションを保存．
 * @version 0.1
 * @date 2023-10-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "cnpy/cnpy.h"
#include "shared/Flow.hpp"
#include "shared/myFunc.hpp"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/matplotlibcpp.h"
namespace plt = matplotlibcpp;
namespace fs = std::filesystem;

// メイン関数
int main() {
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    SMparams params;
    params.nu = 1e-5;
    params.beta = 0.5;
    params.f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 400;
    double dump =0;
    int numThreads = omp_get_max_threads();
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../initials/beta0.5_nu1e-05_15dim_period.npy", true);
    int refresh = 100;
    int candidates = 1000;
    ShellModel SM(params, dt, t_0, t, dump, x_0);
    Eigen::MatrixXcd rawData = npy2EigenMat<std::complex<double>>("../traj/beta0.5_nu1e-05_400period.npy", true);
    
    /*
                                                          █
     ████                         █                       █      █                   ██     ██         ██      █
    ██   █                        █                       █                          ██     ██         ██
    █     █   ███   █ ███         █   ███    ███    ███   █████  █   ███   █ ███     ███    ██   ███  ████ ███ █  █  ██
    █        ██  █  ██  █         █  █  ██  ██  █  ██  █  ██  █  █  █  ██  ██  █     █ █   █ █  █  ██  ██  ██  █  ██ █
    █  ████  █   █  █   █         █      █  █      █   ██ █   ██ █      █  █   █     █ ██  █ █      █  ██  █   █   ███
    █     █  █████  █   █         █   ████  █      █    █ █   ██ █   ████  █   █     █  █ ██ █   ████  ██  █   █   ██
    █     █  █      █   █         █  █   █  █      █   ██ █   ██ █  █   █  █   █     █  █ █  █  █   █  ██  █   █   ███
     █   ██  ██  █  █   █     █  ██  █  ██  ██  █  ██  █  ██  █  █  █  ██  █   █     █   ██  █  █  ██  ██  █   █  ██ █
      ████    ████  █   █     ████   █████   ███    ███   █████  █  █████  █   █     █   █   █  █████   ██ █   █  █  ██
    */
    // パラメータの設定（例）
    int dim = rawData.rows() - 1;
    // データの整形(実関数化)
    Eigen::MatrixXd Data(dim*2, rawData.cols());
    for (int i = 0; i < dim; ++i) {
        Data.row(2*i) = rawData.row(i).real();
        Data.row(2*i+1) = rawData.row(i).imag();
    }
    
    int numTimeSteps = Data.cols();
    int numVariables = Data.rows();
    // 整数の一葉乱数を作成する
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_int_distribution<> dist1(0, numTimeSteps-1);
    
    // ヤコビ行列をcandidates個横に並べたワイドな行列
    Eigen::MatrixXd jacobian_matrix(numVariables, numVariables * candidates);
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic) firstprivate(SM) shared(Data, jacobian_matrix)
    for (int i = 0; i < candidates; ++i){
        // ヤコビアンの計算
        Eigen::MatrixXd jacobian = SM.jacobian_matrix(Data.col(dist1(engine))); // ランダムにデータを選ぶ
        jacobian_matrix.middleCols(i*numVariables, numVariables) = jacobian;
    }
    
    /*
                   █                             █                                     █
     ████          █                      █  ██  █             █                       █      █                   ██     ██         ██      █
    ██   █         █                         ██  █             █                       █                          ██     ██         ██
    █    ██  ███   █   ███     ██  ██  █  █ ████ █████         █   ███    ███    ███   █████  █   ███   █ ███     ███    ██   ███  ████ ███ █  █  ██
    █       █  ██  █  ██  █     █  ██  █  █  ██  ██  █         █  █  ██  ██  █  ██  █  ██  █  █  █  ██  ██  █     █ █   █ █  █  ██  ██  ██  █  ██ █
    █           █  █  █         █  ██  █  █  ██  █   █         █      █  █      █   ██ █   ██ █      █  █   █     █ ██  █ █      █  ██  █   █   ███
    █        ████  █  █         █ █ █ ██  █  ██  █   █         █   ████  █      █    █ █   ██ █   ████  █   █     █  █ ██ █   ████  ██  █   █   ██
    █    ██ █   █  █  █         ███  ██   █  ██  █   █         █  █   █  █      █   ██ █   ██ █  █   █  █   █     █  █ █  █  █   █  ██  █   █   ███
    ██   █  █  ██  █  ██  █      ██  ██   █  ██  █   █     █  ██  █  ██  ██  █  ██  █  ██  █  █  █  ██  █   █     █   ██  █  █  ██  ██  █   █  ██ █
     ████   █████  █   ███       █   ██   █   ██ █   █     ████   █████   ███    ███   █████  █  █████  █   █     █   █   █  █████   ██ █   █  █  ██
    */

    Eigen::MatrixXd traj(dim*2+1, SM.steps + 1);
    
    // ヤコビアンを作用させる用の実ベクトル
    Eigen::VectorXd state = Eigen::VectorXd::Zero(dim*2);
    for (int i = 0; i < dim; ++i) {
        state(2*i) = x_0(i).real();
        state(2*i+1) = x_0(i).imag();
    }
    traj.topLeftCorner(dim*2, 1) = state;
    double time = 0;
    // ヤコビ行列による時間発展
    // ヤコビ行列の選択
    std::uniform_int_distribution<> dist2(0, candidates-1);
    for (int i = 1; i < SM.steps; i++){
        //時間発展
        Eigen::MatrixXd jacobian = jacobian_matrix.middleCols(dist2(engine)*numVariables, numVariables); // ステップごとの
        jacobian = myfunc::regularizeJacobian(jacobian);
        state = myfunc::rungeKuttaJacobian(state, jacobian, dt);
        time += dt;
        traj.col(i) << state, time;
        std::cout << "\r processing..." << i << "/" << traj.cols() << std::flush;
    }

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
    std::vector<std::vector<double>> xs(dim, std::vector<double>()), ys(dim, std::vector<double>());
    long long counter = 0;
    for(int i=0; i < traj.cols(); i++){
        for (int j=0; j < dim; j++){
            xs[j].push_back(traj(2*j, i));
            ys[j].push_back(traj(2*j+1, i));
        }
        
        if (i%refresh == 0){
            plt::clf();
            for (int j=0; j < dim; j++){
                plt::subplot(5, 3, j+1);
                //plot trajectory
                std::map<std::string, std::string> keywords1;
                keywords1.insert(std::make_pair("c", "b")); 
                keywords1.insert(std::make_pair("lw", "1"));
                // keywords1.insert(std::make_pair("alpha", "1"));
                plt::plot(xs[j],ys[j], keywords1);
                plt::xlabel("Real($U_{" + std::to_string(j+1) + "}$)");
                plt::ylabel("Imag($U_{" + std::to_string(j+1) + "}$)");
            }
            std::map<std::string, double> keywords;
            keywords.insert(std::make_pair("hspace", 0.5)); // also right, top, bottom
            keywords.insert(std::make_pair("wspace", 0.5)); // also hspace
            plt::subplots_adjust(keywords);
            std::ostringstream oss;
            oss << "../../animation_frames/jacobi_real-imag_beta" << params.beta << "nu" << params.nu << "_" << std::setfill('0') << std::setw(6) << counter << ".png";
            counter++;
            std::string plotfname = oss.str(); // 文字列を取得する
            plt::save(plotfname);
            oss.str("");
            std::cout << "\r processing..." << i << "/" << traj.cols() << std::flush;
        }
    }
    std::cout << traj(0, 0) << std::endl;

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
    oss << "../../animation/jacobian_real-imag_beta" << params.beta << "nu" << params.nu <<"_"<< t-t_0 << "period.mp4";  // 文字列を結合する
    std::string outputFilename =  oss.str();
    std::string folderPath = "../../animation_frames/"; // フォルダのパスを指定
    int framerate = 10; // フレーム間の遅延時間（ミリ秒）

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