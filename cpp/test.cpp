#include "shared/Eigen_numpy_converter.hpp"
#include <iostream>

int main(){
    // error_vecとerror_rate_vecとtime_vecを一時ファイルにまとめて保存
    std::string data_file = "tmp.dat";
    std::ofstream ofs(data_file);
    for (int i = 0; i < error_vec.size(); i++) {
        ofs << error_vec[i] << "\t" << growth_rate_vec[i] << "\t" << time_vec[i] << std::endl;
    }
    ofs.close();

    // gnuplotでプロット
    std::ostringstream oss;

    // エラー : エラー成長率
    oss << "../../error_growth/error-rate_beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions << "sampling" << sampling_rate << ".png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    std::ostringstream gnuplotCmd;
    gnuplotCmd << "gnuplot -persist -e \"";
    gnuplotCmd << "set term png size 1200,800 font 'Times New Roman,20'; ";
    gnuplotCmd << "set output '" << plotfname << "'; ";
    gnuplotCmd << "set xlabel 'E'; ";
    gnuplotCmd << "set ylabel 'E/E'; ";
    gnuplotCmd << "set logscale x; set format x '10^{%L}';";
    gnuplotCmd << "set autoscale; ";
    gnuplotCmd << "unset key; ";
    gnuplotCmd << "plot '"<< data_file << "' u 1:2 with points pt 7 lc 'blue'; ";
    gnuplotCmd << "replot; ";
    gnuplotCmd << "set output; ";
    gnuplotCmd << "set term qt; ";
    gnuplotCmd << "exit;\"";
    // std::cout << gnuplotCmd.str() << std::endl;
    system(gnuplotCmd.str().c_str());

    // 時間 : エラー
    oss.str("");
    oss << "../../error_growth/time-error_beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions << "sampling" << sampling_rate << ".png";  // 文字列を結合する
    std::string plotfname1 = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname1 << std::endl;
    gnuplotCmd.str("");
    gnuplotCmd << "gnuplot -persist -e \"";
    gnuplotCmd << "set term png size 1200,800 font 'Times New Roman,20'; ";
    gnuplotCmd << "set output '" << plotfname1 << "'; ";
    gnuplotCmd << "set xlabel 't'; ";
    gnuplotCmd << "set ylabel 'E'; ";
    gnuplotCmd << "set logscale y;  set format y '10^{%L}';";
    gnuplotCmd << "unset key; ";
    gnuplotCmd << "plot '"<< data_file << "' u 3:1 with points pt 7 lc 'blue'; ";
    gnuplotCmd << "replot; ";
    gnuplotCmd << "set output; ";
    gnuplotCmd << "set term qt; ";
    gnuplotCmd << "exit;\"";
    // std::cout << gnuplotCmd.str() << std::endl;
    system(gnuplotCmd.str().c_str());

    // 時間 : エラー成長率
    oss.str("");
    oss << "../../error_growth/time-rate_beta" << params.beta << "nu" << params.nu << "t" << t << "dt" << dt << "repeat" << repetitions << "sampling" << sampling_rate << ".png";  // 文字列を結合する
    std::string plotfname2 = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname2 << std::endl;
    gnuplotCmd.str("");
    gnuplotCmd << "gnuplot -persist -e \"";
    gnuplotCmd << "set term png size 1200,800 font 'Times New Roman,20'; ";
    gnuplotCmd << "set output '" << plotfname2 << "'; ";
    gnuplotCmd << "set xlabel 't'; ";
    gnuplotCmd << "set ylabel 'E/E'; ";
    gnuplotCmd << "unset key; ";
    gnuplotCmd << "plot '"<< data_file << "' u 3:2 with points pt 7 lc 'blue'; ";
    gnuplotCmd << "replot; ";
    gnuplotCmd << "set output; ";
    gnuplotCmd << "set term qt; ";
    gnuplotCmd << "exit;\"";
    // std::cout << gnuplotCmd.str() << std::endl;
    system(gnuplotCmd.str().c_str());
}