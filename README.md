# Shell Model研究用コード(C++)

## レイアウト
.
├── README.md このファイル
├── average_time_para 平均ラミナー継続時間
├── beta0.4162_nu0.00018_20000period.npy
├── bif_data 分岐図
├── code コード置き場
    .
    ├── CMakeLists.txt
    ├── LongLaminar.cpp
    ├── Runge_Kutta.hpp
    ├── ShellModel.cpp
    ├── bifurcation.cpp
    ├── build
    ├── cnpy
    ├── dist_of_average_laminar_time.cpp
    ├── dist_of_laminar_time.cpp
    ├── dist_of_max_laminar_time.cpp
    ├── energy_spectrum.cpp
    ├── good_initialvalue.cpp
    ├── laminar_state_before_transition.cpp
    ├── lorenz.cpp
    ├── lyapunov.cpp
    ├── lyapunov_lorenz.cpp
    ├── matplotlib-cpp
    ├── matplotlibcpp.h
    ├── np_load.cpp
    ├── npy_concatenate.cpp
    ├── parameter_serch.cpp
    ├── poincare_section.cpp
    ├── stable_laminar_parameter_map.cpp
    ├── stagger_and_step.cpp
    ├── test.cpp
    ├── trajectory_observation.cpp
    └── who_get_out_laminar_first.cpp
├── distribution 分布データ
├── end_laminar ラミナーが終わる付近のポアンカレ断面
├── generated_lam ラミナーデータ置き場(空)
├── generated_lam_imag ラミナープロット画像
├── initials 初期条件
├── lorenz.npy ローレンツモデルの軌道(削除予定)
├── max_time_para 最長ラミナー継続時間
├── poincare ポアンカレ断面画像
├── stable_para 安定軌道を持つパラメータ
└── traj_images 軌道プロット(1 shellのみ)