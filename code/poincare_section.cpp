#include <iostream>
#include <fstream>
#include <iomanip>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <unordered_set>
#include "Runge_Kutta.hpp"
#include <chrono>
#include <omp.h>
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
Eigen::MatrixXd loc_max(const Eigen::MatrixXd& traj_abs, int obs_dim, double dt);
Eigen::VectorXcd npy2EigenVec(const char* fname);
Eigen::MatrixXcd npy2EigenMat(const char* fname);
Eigen::MatrixXd poincare_section(const Eigen::MatrixXd& traj_abs, int cut_dim, double cut_value);
std::vector<int> extractCommonColumns(const std::vector<Eigen::MatrixXd>& matrices);

int main(){
    auto start = std::chrono::system_clock::now(); // timer start
    double nu = 0.00018;
    double beta = 0.4165;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double ddt = 0.01;
    double t_0 = 0;
    double t = 1000;
    double latter = 2;
    Eigen::VectorXcd x_0 = npy2EigenVec("../../initials/beta0.41616nu0.00018_1.00923e+06period.npy");
    std::vector<Eigen::MatrixXd> matrices; //ポアンカレ写像の結果を格納するベクトル

    ShellModel SM(nu, beta, f, ddt, t_0, t, latter, x_0);
    // 計算する場合は以下のコメントアウトを外す
    // Eigen::MatrixXcd trajectory = SM.get_trajectory_();
    // 計算済みの場合は以下のコメントアウトを外す
    Eigen::MatrixXcd trajectory = npy2EigenMat("../../sync/beta_0.4162nu_0.00018_100000period100000window.npy");
    Eigen::MatrixXd traj_abs = trajectory.cwiseAbs();
    
    Eigen::MatrixXd loc_max_4 = loc_max(traj_abs, 4, ddt);
    matrices.push_back(loc_max_4);
    // Eigen::MatrixXd PoincareSection6 = poincare_section(traj_abs, 6, 0.1);
    // matrices.push_back(PoincareSection6);
    // Eigen::MatrixXd PoincareSection9 = poincare_section(traj_abs, 9, 0.055);
    // matrices.push_back(PoincareSection9);

    // // 共通している列を求める
    std::vector<int> commonColumns = extractCommonColumns(matrices);
    

    std::cout << commonColumns.size() <<"points"<< std::endl; //print the number of points
    plt::figure_size(1000, 1000);
    int plot_dim1 = 3;
    int plot_dim2 = 4;
    // Add graph title
    std::vector<double> x(commonColumns.size()),y(commonColumns.size());
    int ite = 0;
    for (const auto& index :commonColumns) {
            x[ite] = loc_max_4(plot_dim1-1, index);
            y[ite] = loc_max_4(plot_dim2-1, index);
            ite++;
    }
    plt::scatter(x,y);
    std::ostringstream oss;
    oss <<"Shell"<< plot_dim1;
    plt::xlabel(oss.str()); 
    oss.str("");
    oss <<"Shell"<< plot_dim2;
    plt::ylabel(oss.str());
    plt::ylim(0.15, 0.4);
    plt::xlim(0.05, 0.6);
    oss.str("");
    oss << "../../poincare/beta_" << beta << "nu_" << nu << "loc_max_4"<< t / latter <<"period.png";  // 文字列を結合する
    // oss << "../../poincare/beta_" << beta << "nu_" << nu << "loc_max_4_laminar43000period.png";
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    //timer stops
    auto end = std::chrono::system_clock::now();
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

Eigen::MatrixXcd npy2EigenMat(const char* fname){
    std::string fname_str(fname);
    cnpy::NpyArray arr = cnpy::npy_load(fname_str);
    if (arr.word_size != sizeof(std::complex<double>)){
        throw std::runtime_error("Unsupported data type in the npy file.");
    }
    Eigen::Map<const Eigen::MatrixXcd> MatT(arr.data<std::complex<double>>(), arr.shape[1], arr.shape[0]);
    return MatT.transpose();
}

Eigen::MatrixXd loc_max(const Eigen::MatrixXd& traj_abs, int loc_max_dim, double dt){
    // 条件に合えば1, 合わなければ0のベクトルを作成
    std::vector<int> binLoc_max(traj_abs.cols());
    //　最初の3点と最後の3点は条件を満たせないので0
    for (int i = 0; i < 3; ++i){
        binLoc_max[i] = 0;
        binLoc_max[binLoc_max.size()-1-i] = 0;
    }
    for (int i = 0; i < traj_abs.cols()-6; ++i){
        //不連続を除く
        if (traj_abs(traj_abs.rows()-1, i+6) - traj_abs(traj_abs.rows()-1, i) > dt*7){
            binLoc_max[i+3] = 0;
        }
        //極大値か判定
        else if (traj_abs(loc_max_dim, i+1) - traj_abs(loc_max_dim, i) > 0
        && traj_abs(loc_max_dim, i+2) - traj_abs(loc_max_dim, i+1) > 0
        && traj_abs(loc_max_dim, i+3) - traj_abs(loc_max_dim, i+2) > 0
        && traj_abs(loc_max_dim, i+4) - traj_abs(loc_max_dim, i+3) < 0
        && traj_abs(loc_max_dim, i+5) - traj_abs(loc_max_dim, i+4) < 0
        && traj_abs(loc_max_dim, i+6) - traj_abs(loc_max_dim, i+5) < 0){
            binLoc_max[i+3] = 1;
        }
        //
        else{
            binLoc_max[i+3] = 0;
        }
    }
    //binLoc_maxの1の数を数える
    int count = 0;
    for (int i = 0; i < binLoc_max.size(); ++i){
        if (binLoc_max[i] == 1){
            count++;
        }
    }
    Eigen::MatrixXd loc_max_point(traj_abs.rows(),count);
    int col_now = 0;
    for (int i = 0; i < binLoc_max.size(); ++i){
        if (binLoc_max[i] == 1){
            loc_max_point.col(col_now) = traj_abs.col(i);
            col_now++;
        }
    }
    return loc_max_point;
}

Eigen::MatrixXd poincare_section(const Eigen::MatrixXd& traj_abs, int cut_dim, double cut_value){
    // 条件に合えば1, 合わなければ0のベクトルを作成
    std::vector<int> binSection(traj_abs.cols(), 0);

    for (int i = 0; i < traj_abs.cols() -1; ++i){
        if ((traj_abs(cut_dim, i) > cut_value && traj_abs(cut_dim - 1, i+1) < cut_value)
        || (traj_abs(cut_dim - 1, i) < cut_value && traj_abs(cut_dim - 1, i+1) > cut_value)){
            binSection[i] = 1;
            binSection[i+1] = 1;
        }
    }
    //binSectionの1の数を数える
    int count = 0;
    for (int i = 0; i < binSection.size(); ++i){
        if (binSection[i] == 1){
            count++;
        }
    }
    //binSectionの1の数だけの行列を作成
    Eigen::MatrixXd PoincareSection(traj_abs.rows(), count);
    int col_now = 0;
    for (int i = 0; i < binSection.size(); ++i){
        if (binSection[i] == 1){
            PoincareSection.col(col_now) = traj_abs.col(i);
            col_now++;
        }
    }
    return PoincareSection;
}

std::vector<int> extractCommonColumns(const std::vector<Eigen::MatrixXd>& matrices) {
    // ベースとなる行列を選ぶ（ここでは最初の行列）
    const Eigen::MatrixXd& baseMatrix = matrices.front();
    
    std::vector<int> commonColumns; // 共通の列のBase行列におけるインデックスを格納する配列
    // matricesの要素が1つの場合、Base行列の全てのインデックスを格納して返す
    if (matrices.size() == 1) {
        for (int i = 0; i < baseMatrix.cols(); ++i) {
            commonColumns.push_back(i);
        }
        return commonColumns;
    }
    // ベース行列の各列を走査
    for (int baseCol = 0; baseCol < baseMatrix.cols(); ++baseCol) {
        bool isCommon = true; // 共通の列があるかを示すフラグ

        // ベース行列の現在の列と同じ列が他の行列にあるか検証
        for (size_t i = 1; i < matrices.size(); ++i) {
            //フラグがfalse、つまり既に共通の列がない場合はループを抜ける
            if (!isCommon) {
                break;
            }
            const Eigen::MatrixXd& matrix = matrices[i];
            isCommon = false;
            // 共通の列があればtrue,　なければfalse
            for (int col=0; col < matrix.cols(); ++col) {
                // ベース行列の現在の列と他の行列の現在の列が等しいかを比較
                if (baseMatrix.col(baseCol) == matrix.col(col)) {
                    isCommon = true;
                    break;
                }
            }
        }

        // 共通の列であればインデックスを追加
        if (isCommon) {
            commonColumns.push_back(baseCol);
        }
    }

    return commonColumns;
}