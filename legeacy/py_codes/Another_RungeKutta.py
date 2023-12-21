from numba import njit
import numpy as np


"""
長時間ラミナー作成用クラス
"""    
class Long_laminar:
    def __init__(self):
        self.arr = None
        self.average = None

    #微分方程式を定義
    @staticmethod
    @njit(cache=True)
    #@njit(c16[:,:](f8,c16[:,:],f8,c16,f8[:,:],f8[:,:],f8[:,:],f8[:,:]),cache=True)
    def goy_shell_model(t,value, nu, f, k_n, c_n_1, c_n_2, c_n_3):
        delta = np.zeros((len(value),1), dtype = np.complex128)
        delta[0,0:] = f

        u = np.zeros((len(value)+4, 1), dtype = np.complex128)
        u[2:-2,0:]=value #uは両端2個は0,その間にu1~unが入ってる縦array

        #微分方程式
        ddt_u = (c_n_1 * np.conj(u[3:-1,0:]) * np.conj(u[4:,0:]) + c_n_2 * np.conj(u[1:-3,0:]) * np.conj(u[3:-1,0:]) + c_n_3 * np.conj(u[1:-3,0:]) * np.conj(u[:-4,0:])) * 1j + delta - nu * u[2:-2] * (k_n[2:-2] ** 2)

        return ddt_u
    
    #4段4次陽的ルンゲクッタ
    @staticmethod
    @njit(cache=True)
    def rk4(x,y,step,nu, f, k_n, c_n_1, c_n_2, c_n_3,func):
        #xは時間、yは前の値。
        k1 = step * func(x, y, nu, f, k_n, c_n_1, c_n_2, c_n_3)
        k2 = step * func(x + step / 2, y + k1 / 2, nu, f, k_n, c_n_1, c_n_2, c_n_3)
        k3 = step * func(x + step / 2, y + k2 / 2, nu, f, k_n, c_n_1, c_n_2, c_n_3)
        k4 = step * func(x + step / 2, y + k3, nu, f, k_n, c_n_1, c_n_2, c_n_3)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    
    #摂動()を与える　*時間は入れない*
    @staticmethod
    @njit(cache=True)
    def perturbator(array):
        a = -3
        b = -14
        s = (b - a) * np.random.rand() + a
        
        u = array * (2 * np.random.rand(len(array[:,0]),1) - 1)
        
        u_scaled = u / np.linalg.norm(u)
        
        return u_scaled * (10 ** s) + array
    
    @staticmethod
    @njit(cache=True)
    def judge_laminar_for_a_while(nu, f, k_n, c_n_1, c_n_2, c_n_3, step, rk4, func, check_sec, progress_sec, start, laminar,skip, epsilon, judge):
        """ 
        ある時点から指定秒数走らせて、ラミナーに入っているかどうか指定秒を返す
        """ 
        step_number = int((check_sec+1E-10)//step)
        zero = np.zeros((len(start[:,0]), int((progress_sec+1E-10)//step)+1), dtype = np.float64)
        zero = zero.astype(np.complex128)
        arr = np.zeros((len(start),step_number+1),dtype=np.float64) #入れ物を作る
        arr = arr.astype(np.complex128) #複素数である場合は実行
        t = start[-1,0]
        arr[:,0:1] = start
        for i in range(step_number): #ここのfor文でルンゲクッタ実行
            arr[:-1,i+1:i+2]=rk4(t,arr[:-1,i:i+1], step, nu, f, k_n, c_n_1, c_n_2, c_n_3, func)
            t+=step
            arr[-1,i+1]=t
            if i % skip == 0:
                if judge(arr[:-1,i:i+1], laminar, epsilon) == 0:
                    return zero
        return arr[:,:int((progress_sec+1E-10)//step)+1].copy()
                    
        
    
    @staticmethod
    #@njit
    def get_laminar_wrapped(nu, f, k_n, c_n_1, c_n_2, c_n_3, start, step, period, rk4, function, judge_laminar_for_a_while, perturbator, judge, check_sec, progress_sec, laminar, skip, epsilon):
        dimention = len(start) #変数の次元
        step_number = abs(int((period[1] - period[0]+1E-10) // step)) #刻む回数
        
        arr = np.zeros((dimention+1,step_number+1),dtype = np.float64) #器を作る(1列のみ)
        arr = arr.astype(np.complex128) #複素数の時は実行
        arr[:-1,0:1] = start
        arr[-1,0] = period[0]
        i=0 #イテレーター
        perturbated_time = []
        cycle_limit = 3E+4
        while abs(arr[-1,-1]) == 0:
            if round(abs(arr[-1,i])) % 10 == 0:
                print(f'{round(abs(arr[-1,i]))}時間', end = '') #時間ごとにプリント
            
            dummy = judge_laminar_for_a_while(nu, f, k_n, c_n_1, c_n_2, c_n_3, step, rk4, function, check_sec, progress_sec, arr[:,i:i+1], laminar,skip, epsilon, judge)
            if not dummy[0,0] == 0:
                arr[:,i:i+len(dummy[0,:])] = dummy
                i += len(dummy[0,:]) - 1 #イテレータを次の初期値(今回の最後)まで持ってく
            else:
                perturbated_time.append(abs(arr[-1,i]))
                n = 1
                while True:
                    #摂動を加えた点pertubatedを作る
                    perturbated = arr[:,i:i+1].copy() #最終地点をコピってくる
                    perturbated[:-1,:] = perturbator(perturbated[:-1,:]) #摂動を与える
                    dummy = judge_laminar_for_a_while(nu, f, k_n, c_n_1, c_n_2, c_n_3, step, rk4, function, check_sec, progress_sec, perturbated, laminar, skip, epsilon, judge)
                    if n % 100 == 0:
                        if n == 100:
                            print(f'\n{n}試行目', end='')
                        print(f'\r {n}試行目', end='')
                    n += 1
                    if n > cycle_limit:
                        return arr[:,:i+1], perturbated_time #ダメだったらゼロ行列を返す(numbaを使う上でreturnの型を統一しなければならない)
                    
                    if not dummy[0,0] == 0:
                        arr[:,i:i+len(dummy[0,:])] = dummy
                        i += len(dummy[0,:]) - 1 #イテレータを次の初期値(今回の最後)まで持ってく
                        if n > cycle_limit*0.8:
                            cycle_limit *=1.5
                        break
        return arr, perturbated_time
    
    def get_laminar(self, nu, f, k_n, c_n_1, c_n_2, c_n_3, start, step, period, check_sec, progress_sec, laminar, skip, epsilon):
        self.arr, self.perturbated_time = Long_laminar.get_laminar_wrapped(nu, f, k_n, c_n_1, c_n_2, c_n_3, start, step, period, Long_laminar.rk4, Long_laminar.goy_shell_model, Long_laminar.judge_laminar_for_a_while, Long_laminar.perturbator, Long_laminar.judge_in_laminar_or_not, check_sec, progress_sec, laminar, skip, epsilon)
        
    @staticmethod
    @njit(nogil=True, cache=True)
    def judge_in_laminar_or_not(point, laminar, epsilon):
        a = (np.abs(laminar[:4,:]) - np.abs(point[:4,:]))**2 #the distance between point and laminar is mesured in absolute space in order to reduce the process
        b = np.real(np.sum(a, axis = 0) ** (1/2))
        if b[np.where(b < epsilon)].shape[0] == 0:
            return 0
        else:
            return 1
