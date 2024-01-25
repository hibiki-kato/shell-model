#include "myFunc.hpp"
#include <iostream>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <vector>
#include <unordered_set>

namespace myfunc{
    std::string ordinal_suffix(int n) {
    return std::to_string(n) + (n % 100 < 11 || n % 100 > 13 ? (n % 10 == 1 ? "st" : (n % 10 == 2 ? "nd" : (n % 10 == 3 ? "rd" : "th"))) : "th");
    }
    void duration(std::chrono::time_point<std::chrono::system_clock> start){
        auto end = std::chrono::system_clock::now(); // 計測終了時間
        int hours = std::chrono::duration_cast<std::chrono::hours>(end-start).count(); //処理に要した時間を変換
        int minutes = std::chrono::duration_cast<std::chrono::minutes>(end-start).count(); //処理に要した時間を変換
        int seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count(); //処理に要した時間を変換
        int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間を変換
        std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;
    }
    int count_mesh(const Eigen::MatrixXd& trajectory, int n, std::vector<std::tuple<int, double, double>> intervals){
        //trajectoryをrowmajorに変換する(tall matrix)
        Eigen::MatrixXd trajectory_T = trajectory.transpose();
        //各点がどのメッシュに含まれるかを記録する(tall matrix)
        Eigen::MatrixXi index(trajectory_T.rows(), intervals.size());

        //intervalsを0,1に正規化する写像は，(x - min) / (max - min)である
        //trajectoryの各点を写像して，各次元の値を0,1に正規化し，n倍して，整数にする(切り捨て0~n-1)
        for(int i=0;i<intervals.size();i++){
            int var = std::get<0>(intervals[i]);
            double min = std::get<1>(intervals[i]);
            double max = std::get<2>(intervals[i]);
            for(int j=0;j<trajectory_T.rows();j++){
                index(j, i) = static_cast<int>((trajectory_T(j, var) - min) / (max - min) * n);
            }
        }

        //unordered_setを使って，点が含まれる点の個数を数える
        std::unordered_set<std::vector<int>, VectorHash> index_set;
        for(int i=0;i<index.rows();i++){
            std::vector<int> tmp(index.cols());
            for(int j=0;j<index.cols();j++){
                tmp[j] = index(i, j);
            }
            index_set.insert(tmp);
        }
        return index_set.size();
    }
}