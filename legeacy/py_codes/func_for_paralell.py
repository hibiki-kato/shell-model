import numpy as np
from tqdm import tqdm
from numba import njit
import time
from matplotlib import pyplot as plt

import sys
sys.path.append('../../lb')
import RungeKutta_acceralated

def wait_a_seconds(a):
    print(f"{a} is started")
    time.sleep(a)
    print(f"{a}is completed")
    return a

def competition(runge_para, competitives, nu = True, start = True):
    #摂動を与えたnuのリスト作成
    lmodel = RungeKutta_acceralated.Long_laminar()
    model = RungeKutta_acceralated.RungeKutta()
    
    if nu is True:
        nus = np.zeros((competitives),dtype = np.float64)
        nus[0] = runge_para["nu"]*1
        for i in range(competitives-1):
            a=-4.2
            b=-10
            nus[i+1] = runge_para["nu"] + (10) ** ((b - a) * np.random.rand() + a) * (-1) ** np.random.randint(1,3) 
    else:
        nus = np.zeros((competitives),dtype = np.float64) + runge_para["nu"]
    if start is True:
        #摂動を与えた初期値のリスト作成
        starts = np.zeros((14,competitives),dtype = np.complex128)
        starts[:,0:1] = runge_para["start"].copy()
        for i in range(competitives-1):
            starts[:,i+1:i+2] = lmodel.perturbator(runge_para['start'])
            
    else:
        starts = np.zeros((14,competitives),dtype = np.complex128)
        for i in range(competitives):
            starts[:,i:i+1] = runge_para['start']
            
    #ラミナーから外れた時間
    times = np.zeros((competitives)) + runge_para["period"][1]
    #U_tra_m作成
    #Trajectie matrix
    w=60
    m=3
    laminar=abs(np.load("../laminar_initials/beta0.423_nu0.00017584784643038092_step0.01_1500check_500pro_20000period.npz")["laminar"][0,::500])
    tra_matrix = np.zeros((w, len(laminar) - w+1))
    for j in range(tra_matrix.shape[1]):
        tra_matrix[:, j] = laminar[j:j+w] * 1
    U_tra, _, _  = np.linalg.svd(tra_matrix, full_matrices=False)
    U_tra_m  = U_tra[:, :m]

    for i in range(competitives):
        runge_para['start'] = starts[:,i:i+1].copy()
        runge_para['nu'] = nus[i]
        #インスタンス化&実行
        model.get_arr_latter(**runge_para,latter=1)

        #極大抽出
        loc_max = lmodel.loc_max_7(model.arr_latter, 1) #u1が極大値なものを拾ってくる
        loc_min = lmodel.loc_min_7(model.arr_latter, 1) #u1が極小値なものを拾ってくる

        #極大値で判定
        for j in range(loc_max.shape[1]):
            if not 0.87 < loc_max[0,j] < 1.1:
                times[i] = loc_max[-1, j]
                break
        #極小値で判定
        for j in range(loc_min.shape[1]):
            if not 0.5 < loc_min[0,j] < 0.64:
                times[i] = min([loc_min[-1,j], times[i]])
                break
        
        # #異常値検知
        
        
        results = RungeKutta_acceralated.Long_laminar.sst(np.abs(model.arr_latter[0,::500].T), w, U_tra_m = U_tra_m)
        for j in range(results.shape[0]):
            if results[j] > 0.0008:
                times[i] = min(np.abs(model.arr_latter[-1,::500][j]), times[i])
                break
        
        # for j in range(model.arr_latter.shape[1]):
        #     if not (
        #         (0.48 < np.abs(model.arr_latter[0, j])) and np.max(np.abs(model.arr_latter[0, j]) < 0.55)
        #         and (0.18 < np.abs(model.arr_latter[1, j]) and np.abs(model.arr_latter[1, j]) < 0.27)
        #         and (0.19 < np.abs(model.arr_latter[2, j]) and np.abs(model.arr_latter[2, j]) < 0.33)
        #         and (0.1 < np.abs(model.arr_latter[3, j]) and np.abs(model.arr_latter[3, j]) < 0.3)
        #         and (0 < np.abs(model.arr_latter[4, j]) and np.abs(model.arr_latter[4, j]) < 0.28)
        #         and (0.06 < np.abs(model.arr_latter[5, j]) and np.abs(model.arr_latter[5, j]) < 0.18)
        #         and (0 < np.abs(model.arr_latter[6, j]) and np.abs(model.arr_latter[6, j]) < 0.2)
        #         and (0.01 < np.abs(model.arr_latter[7, j]) and np.abs(model.arr_latter[7, j]) < 0.14)
        #         and (0 < np.abs(model.arr_latter[8, j]) and np.abs(model.arr_latter[8, j]) < 0.066)
        #         and (0 < np.abs(model.arr_latter[13, j]) and np.abs(model.arr_latter[13, j]) < 0.000035)
        #     ):
        #         times[i] = np.abs(model.arr_latter[-1, j])
            
    runge_para['nu'] = nus[np.argmax(times)]
    runge_para['start'] = starts[:, np.argmax(times):np.argmax(times)+1]
    #print(max(times))
    return times, starts, nus

def perturbator(array):
    a = -2
    b = -7
    s = (b - a) * np.random.rand() + a
    
    u = array * (2 * np.random.rand(len(array[:,0]),1) - 1)
    
    u_scaled = u / np.linalg.norm(u)
    
    return u_scaled * (10 ** s) + array
    

def laminar_distribution(runge_para, laminar, skip, epsilon):
    runge_para["start"] = perturbator(runge_para["start"])
    model = RungeKutta_acceralated.RungeKutta()
    model.get_laminar_time(**runge_para,laminar=laminar,skip=skip,epsilon = epsilon)
    
    binary_array = model.Binary[0,:]
    # バイナリ配列を1が現れるたびに分割
    split_array = np.split(binary_array, np.where(binary_array == 0)[0])

    # 分割された部分について1が何個分続いているかを計算
    counts = np.array([len(np.where(sub_array == 1)[0]) for sub_array in split_array if len(sub_array) > 0]) * runge_para["step"] * skip
    counts_no_zero = counts[np.where(np.array(counts) > 0)]
    
    return counts_no_zero

def competition2(runge_para, competitives, laminar, skip, epsilon, nu = True, start = True):
    #摂動を与えたnuのリスト作成
    lmodel = RungeKutta_acceralated.Long_laminar()
    model = RungeKutta_acceralated.RungeKutta()
    
    #make nu list
    if nu is True:
        nus = np.zeros((competitives),dtype = np.float64)
        nus[0] = runge_para["nu"]*1
        for i in range(competitives-1):
            a=-4.2
            b=-10
            nus[i+1] = runge_para["nu"] + (10) ** ((b - a) * np.random.rand() + a) * (-1) ** np.random.randint(1,3) 
    else:
        nus = np.zeros((competitives),dtype = np.float64) + runge_para["nu"]
    
    #make start list                                                                  ]
    if start is True:
        #摂動を与えた初期値のリスト作成
        starts = np.zeros((14,competitives),dtype = np.complex128)
        starts[:,0:1] = runge_para["start"].copy()
        for i in range(competitives-1):
            starts[:,i+1:i+2] = lmodel.perturbator(runge_para['start'])
            
    else:
        starts = np.zeros((14,competitives),dtype = np.complex128)
        for i in range(competitives):
            starts[:,i:i+1] = runge_para['start']
            
    #ラミナーから外れた時間
    times = np.zeros((competitives)) + runge_para["period"][1]
    for i in range(competitives):
        runge_para['start'] = starts[:,i:i+1].copy()
        runge_para['nu'] = nus[i]
        #インスタンス化&実行
        model.get_arr_latter(**runge_para,latter=1)
        times[i] = max_time(model.arr_latter, laminar, skip, epsilon)
            
    runge_para['nu'] = nus[np.argmax(times)]
    runge_para['start'] = starts[:, np.argmax(times):np.argmax(times)+1]
    #print(max(times))
    return times, starts, nus

@njit(nogil = True, cache=True)
def max_time(arr, laminar, skip, epsilon):
    #2nd shell から 4th shellだけで距離を算出
    for i in range(round(arr.shape[1]/skip)):
        a = np.abs(laminar[1:4]) - np.abs(arr[1:4,skip*i:skip*i+1])
        a = a * a
        b = np.sum(a, axis = 0)
        if b[np.where(b < epsilon**2)].shape[0] == 0:
            return np.abs(arr[-1,skip*i])
    return np.abs(arr[-1,-1])

def animation_per_parameter(runge_para, betas, nus, param_num, latter, skip, shell1, shell2):
    model = RungeKutta_acceralated.RungeKutta()
    ims = []
    fig, ax = plt.subplots()
    ax.set_xlabel(f"u{shell1}")
    ax.set_ylabel(f"u{shell2}")
    
    for i in tqdm(range(param_num)):
        #nu更新
        runge_para["nu"] = nus[i]
        beta = betas[i]
        #betaの値に従いc_n_2,c_n_3を更新
        #c_n_2
        runge_para["c_n_2"] = runge_para["k_n"][1:-3, 0:] * -beta
        runge_para["c_n_2"][-1, 0:] = np.zeros((1, 1))
        #c_n_3
        runge_para["c_n_3"] = runge_para["k_n"][:-4, 0:] * (beta - 1)
        
        model.get_arr_latter(**runge_para,latter=latter)
        
        text = ax.text(0.5,1.05,(f"beta={betas[i]} \n nu={nus[i]}"), size = 15, color = "green", ha = "center", transform=ax.transAxes)
    
        im1 = ax.plot(abs(model.arr_latter[shell1-1,::skip]),abs(model.arr_latter[shell2-1,::skip]),color="b",linewidth=0.1)
        ims.append(im1 +[text])
    

    return fig, ims
 
