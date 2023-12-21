import sys
sys.dont_write_bytecode = True
from numba import njit, f8, c16, i8
import numba
import numpy as np
import cmath

class RungeKutta:
    def __init__(self):
        self.arr = None
        self.arr_latter = None
        self.average = None

    #微分方程式を定義
    @staticmethod
    @njit(nogil = True)
    #@njit(c16[:,:](f8,c16[:,:],f8,c16,f8[:,:],f8[:,:],f8[:,:],f8[:,:]),nogil = True)
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
    @njit(nogil = True)
    def rk4(x,y,step,nu, f, k_n, c_n_1, c_n_2, c_n_3,func):
        #xは時間、yは前の値。
        k1 = step * func(x, y, nu, f, k_n, c_n_1, c_n_2, c_n_3)
        k2 = step * func(x + step / 2, y + k1 / 2, nu, f, k_n, c_n_1, c_n_2, c_n_3)
        k3 = step * func(x + step / 2, y + k2 / 2, nu, f, k_n, c_n_1, c_n_2, c_n_3)
        k4 = step * func(x + step / 2, y + k3, nu, f, k_n, c_n_1, c_n_2, c_n_3)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    #ルンゲクッタ実行機
    @staticmethod
    @njit(nogil = True)
    def get_arr_wrapped(arr, nu, f, k_n, c_n_1, c_n_2, c_n_3,start,step,period,rk4,func):
        step_number = abs(int((period[1] - period[0]) // step))

        t = period[0]
        arr[:-1,0:] = start
        arr[-1,0:] = t

        for i in range(step_number): #ここのfor文でルンゲクッタ実行
            arr[:-1,i+1:i+2]=rk4(t,arr[:-1,i:i+1],step,nu, f, k_n, c_n_1, c_n_2, c_n_3,func)
            t+=step
            arr[-1,i+1]=t

        return arr

    #フロントサイド
    def get_arr(self, nu, f, k_n, c_n_1, c_n_2, c_n_3,start,step,period):
        if isinstance(start, np.ndarray):
            dimention = len(start)
        else:
            dimention = 1

        step_number = abs(int((period[1] - period[0]) // step)) #刻む回数
        self.arr = np.zeros((dimention+1,step_number+1),dtype = np.float64)
        self.arr = self.arr.astype(np.complex128)

        self.arr = RungeKutta.get_arr_wrapped(self.arr, nu, f, k_n, c_n_1, c_n_2, c_n_3,start,step,period,RungeKutta.rk4,RungeKutta.goy_shell_model)
    
    #後半までを捨てるルンゲクッタ実行機
    @staticmethod
    @njit(nogil = True)
    def get_arr_latter_wrapped(arr, nu, f, k_n, c_n_1, c_n_2, c_n_3,start,step,period,rk4,func,latter):
        step_number = abs(int((period[1] - period[0]) // step))

        t = period[0]
        arr[:-1,0:] = start
        arr[-1,0:] = t
        
        kohan_stepnum = int(step_number/latter)
        kohan_start = int(int(period[1] - period[0]) - int(period[1] - period[0])/latter) #必要な部分の開始時間
        
        while abs(arr[-1,0]) < kohan_start:
            arr[:-1,0:1] = rk4(t,arr[:-1,0:1],step,nu, f, k_n, c_n_1, c_n_2, c_n_3,func)
            t += step
            arr[-1,0] = t
            
        for i in range(kohan_stepnum):
            arr[:-1,i+1:i+2]=rk4(t,arr[:-1,i:i+1],step,nu, f, k_n, c_n_1, c_n_2, c_n_3,func)
            t+=step
            arr[-1,i+1]=t
            
        return arr
                          
    #フロント            
    def get_arr_latter(self, nu, f, k_n, c_n_1, c_n_2, c_n_3,start,step,period,latter=10):
        if isinstance(start,np.ndarray):
            dimention = len(start)
        else:
            dimention = 1

        step_number = abs(int((period[1] - period[0]) // step)) #刻む回数
        kohan_stepnum = int(step_number/latter) #後半の一部のステップ数
        
        self.arr_latter = np.zeros((dimention+1,kohan_stepnum+1),dtype = np.float64)
        self.arr_latter = self.arr_latter.astype(np.complex128)

        self.arr_latter = RungeKutta.get_arr_latter_wrapped(self.arr_latter, nu, f, k_n, c_n_1, c_n_2, c_n_3,start,step,period,RungeKutta.rk4,RungeKutta.goy_shell_model,latter=latter)    
    
    @staticmethod
    @njit(nogil = True)
    def get_sum_wrapped(nu, f, k_n, c_n_1, c_n_2, c_n_3,start,step,period,rk4,func,latter):
        step_number = np.abs(int((period[1] - period[0]) // step))
        print(step_number)
        dimention = len(start)
        arr = np.zeros((dimention+1,1),dtype = np.complex128)
        t = period[0]
        arr[:-1,0:] = start 
        arr[-1,0:] = t
        sum=np.zeros((dimention,1),dtype = np.float64)
        
        for i in range(step_number): #ここのfor文でルンゲクッタ実行
            arr[:-1,:]=rk4(t,arr[:-1,:],step,nu, f, k_n, c_n_1, c_n_2, c_n_3,func)
            t+=step
            arr[-1,:]=t
            
            if t >= (period[1]-period[0])*(1-1/latter):
                sum += np.abs(arr[:-1,0:])
                
        return sum/(step_number/latter)


    def get_sum(self, nu, f, k_n, c_n_1, c_n_2, c_n_3,start,step,period,latter=1):
        self.average = RungeKutta.get_sum_wrapped(nu, f, k_n, c_n_1, c_n_2, c_n_3,start,step,period,RungeKutta.rk4,RungeKutta.goy_shell_model, latter)
    
    @staticmethod
    @njit(nogil=True, cache=True)
    def judge_in_laminar_or_not(point, laminar, epsilon):
        a = np.abs(laminar[:4,:]) - np.abs(point[:4,:]) #the distance between point and laminar is mesured in absolute space in order to reduce the process
        a = a * a
        b = np.real(np.sum(a, axis = 0) ** (1/2))
        if b[np.where(b < epsilon)].shape[0] == 0:
            return 0
        else:
            return 1
    
    @staticmethod
    @njit(nogil = True)
    def get_laminar_time_wrapped(nu, f, k_n, c_n_1, c_n_2, c_n_3,start,step,period,rk4,func,judge,laminar,skip,epsilon):
        step_number = np.abs(round((period[1] - period[0]) / step))
        dimention = len(start)
        arr = np.zeros((dimention+1,1),dtype = np.complex128)
        t = period[0]
        arr[:-1,:] = start
        arr[-1:,:] = t
        time=np.zeros((1,int(step_number/skip)),dtype = np.int16)
        
        for i in range(step_number): #ここのfor文でルンゲクッタ実行
            arr[:-1,:]=rk4(t,arr[:-1,:],step,nu, f, k_n, c_n_1, c_n_2, c_n_3,func)
            t+=step
            arr[-1:,:]=t
            if i % skip == 0:
                time[0,round(i/skip)] = judge(arr[:-1,:], laminar, epsilon)
        return time
    
    def get_laminar_time(self, nu, f, k_n, c_n_1, c_n_2, c_n_3,start,step,period,laminar,skip,epsilon):
        self.Binary = RungeKutta.get_laminar_time_wrapped(nu, f, k_n, c_n_1, c_n_2, c_n_3,start,step,period,RungeKutta.rk4,RungeKutta.goy_shell_model,RungeKutta.judge_in_laminar_or_not,laminar,skip,epsilon)
    
    
    
"""
長時間ラミナー作成用クラス
"""    
class Long_laminar:
    def __init__(self):
        self.arr = None
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
        b = -10
        s = (b - a) * np.random.rand() + a
        
        u = array * (2 * np.random.rand(len(array[:,0]),1) - 1)
        
        u_scaled = u / np.linalg.norm(u)
        
        return u_scaled * (10 ** s) + array
    
    
    #c_dimが極大値を取る時の全変数(+時間)をndarrayで返す関数
    @staticmethod
    @njit(cache=True)
    def loc_max_7(arr, dim):
        N = dim - 1
        z_list = np.zeros((len(arr[:,0]),1))
        for j in range(len(arr[0,:]) - 3):
            #7点とって極大判定
            if (np.abs(arr[N,j+1]) - np.abs(arr[N,j]) > 0
            and np.abs(arr[N,j+2]) - np.abs(arr[N,j+1]) > 0
            and np.abs(arr[N,j+3]) - np.abs(arr[N,j+2]) > 0
            and np.abs(arr[N,j+4]) - np.abs(arr[N,j+3]) < 0
            and np.abs(arr[N,j+5]) - np.abs(arr[N,j+4]) < 0
            and np.abs(arr[N,j+6]) - np.abs(arr[N,j+5]) < 0):
                z_list = np.append(z_list,np.abs(arr[:,j+3:j+4]),axis=1)
        return z_list[:,1:]
    
    #c_dimが極小値を取る時の全変数(+時間)をndarrayで返す関数
    @staticmethod
    @njit(cache=True)
    def loc_min_7(arr, dim):
        N = dim - 1
        z_list = np.zeros((len(arr[:,0]),1))
        for j in range(len(arr[0,:]) - 3):
            #7点とって極大判定
            if (np.abs(arr[N,j+1]) - np.abs(arr[N,j]) < 0
            and np.abs(arr[N,j+2]) - np.abs(arr[N,j+1]) < 0
            and np.abs(arr[N,j+3]) - np.abs(arr[N,j+2]) < 0
            and np.abs(arr[N,j+4]) - np.abs(arr[N,j+3]) > 0
            and np.abs(arr[N,j+5]) - np.abs(arr[N,j+4]) > 0
            and np.abs(arr[N,j+6]) - np.abs(arr[N,j+5]) > 0):
                z_list = np.append(z_list,np.abs(arr[:,j+3:j+4]),axis=1)
        return z_list[:,1:]
    
    @staticmethod
    @njit(cache=True, nogil = True)
    def flux(arr, N, beta):
        """
        the total flux of energy, In, through the Nth shell (1<N<13 ;int)
        """
        k_N = (2 ** (-4)) ** N
        return np.imag(k_N * arr[N-1:N, :] * arr[N:N+1, :] * (arr[N+1:N+2, :] + (1 - beta)/2 * arr[N-2:N-1, :]))
    

    # @njit(cache = True, nogil = True)
    # def sst(data, w, m=2, k=None, L=None):
    #     """
    #     Parameters
    #     ----------
    #     data : array_like
    #         Input array or object that can be converted to an array.
    #     w    : int
    #         Window size
    #     m    : int
    #         Number of basis vectors
    #     k    : int
    #         Number of columns for the trajectory and test matrices
    #     L    : int
    #         Lag time

    #     Returns
    #     -------
    #     Numpy array contains the degree of change.()
    #     """
    #     # Set variables
    #     if k is None:
    #         k = w // 2
    #     if L is None:
    #         L = k // 2
    #     T = len(data)

    #     # Calculation range
    #     start_cal = k + w
    #     end_cal = T - L + 1

    #     # Calculate the degree of change
    #     change_scores = np.zeros(len(data))
    #     for t in range(start_cal, end_cal + 1):
    #         # Trajectory matrix
    #         start_tra = t - w - k + 1
    #         end_tra = t - w
            
    #         tra_matrix = np.zeros((w, end_tra - start_tra + 1))
    #         i = 0
    #         for j in range(start_tra, end_tra+1):
    #             tra_matrix[:, i] = data[j-1:j-1+w] * 1
    #             i += 1

    #         # Test matrix
    #         start_test = start_tra + L
    #         end_test = end_tra + L
    #         test_matrix = np.zeros((w, end_test - start_test + 1))
    #         i = 0
    #         for j in range(start_test, end_test+1):
    #             test_matrix[:, i] = data[j-1:j-1+w] * 1
    #             i += 1

    #         # Singular value decomposition(SVD)
    #         U_tra, _, _  = np.linalg.svd(tra_matrix, full_matrices=False)
    #         U_test, _, _ = np.linalg.svd(test_matrix, full_matrices=False)
    #         U_tra_m  = U_tra[:, :m:1]
    #         U_test_m = U_test[:, :m:1]
    #         s = np.linalg.svd(np.dot(U_tra_m.T.copy(), U_test_m.copy()), full_matrices=False)[1]
    #         change_scores[t] = 1 - s[0]

    #     return change_scores
    
    @njit(cache=True, nogil=True)
    def sst(data, w,laminar, m=3,  k=None, L=None, U_tra_m=None):
        """_summary_

        Args:
            data (np.ndarray): _description_
            w (int): _description_
            m (int, optional): _description_. Defaults to 3.
            k (int, optional): _description_. Defaults to None.
            L (int, optional): _description_. Defaults to None.
            laminar (np.ndarray, optional): _description_. Defaults to abs(np.load("../laminar_initials/beta0.423_nu0.00017584784643038092_step0.01_1500check_500pro_20000period.npz")["laminar"][0,::500]).
            U_tra_m (np.ndarray, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # Set variables
        if k is None:
            k = w // 2
        if L is None:
            L = k // 2
        T = data.shape[0] # type: ignore

        # Calculation range
        start_cal = 1
        end_cal = T - (k + w) + 1
        if U_tra_m is None:
            #Trajectie matrix
            tra_matrix = np.zeros((w, len(laminar) - w+1))
            for j in range(tra_matrix.shape[1]):
                tra_matrix[:, j] = laminar[j:j+w] * 1
            U_tra, _, _  = np.linalg.svd(tra_matrix, full_matrices=False)
            U_tra_m  = U_tra[:, :m]


        # Calculate the degree of change
        change_scores = np.zeros(data.shape[0]) # type: ignore
        for t in range(start_cal, end_cal + 1):
            # Test matrix
            start_test = t
            end_test = t + k - 1
            test_matrix = np.zeros((w, end_test - start_test + 1))
            i = 0
            for j in range(start_test, end_test+1):
                test_matrix[:, i] = data[j-1:j-1+w] * 1 # type: ignore
                i += 1

            # Singular value decomposition(SVD)
            U_test, _, _ = np.linalg.svd(test_matrix, full_matrices=False)
            U_test_m = U_test[:, :m]
            s = np.linalg.svd(np.dot(U_tra_m.T.copy(), U_test_m.copy()), full_matrices=False)[1]
            change_scores[t-1] = 1 - s[0]

        return change_scores
    
    @staticmethod
    @njit(cache=True, nogil = True)
    def diff6(arr, h):
        """
        離散値において導関数を近似(6次)
        1回微分{f(x+3h)-9f(x+2h)+45f(x+h)-45f(x-h)+9f(x-2h)-f(x-3h)}/60h
        xは時系列データ, hはデータ間の時間(second)
        """
        arr = np.abs(arr)
        res = (arr[:, 6:] - 9*arr[:, 5:-1] + 45*arr[:, 4:-2] - 45*arr[:, 2:-4] + 9*arr[:, 1:-5] - arr[:, :-6]) / (60*h)
        res[-1:, :] = arr[-1:, 3:-3] 
        return res

    
    @staticmethod
    @njit(cache=True)
    def judge_laminar_for_a_while(nu, f, k_n, c_n_1, c_n_2, c_n_3, step, rk4, func, loc_max_7, loc_min_7, flux, check_sec, progress_sec, start):
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
        
        #ポアンカレ断面
        loc_max_1st = loc_max_7(arr, 1) #u1が極大値なものを拾ってくる
        loc_min_1st = loc_min_7(arr, 1) #u1が極小値なものを拾ってくる
        
        loc_max_4th = loc_max_7(arr, 4) #u4が極大値なものを拾ってくる
        beta = -c_n_2[4,0] / k_n[5,0] #fluxのためにbetaを求める
        flux_of_3th = flux(arr, 3, beta) 
        
        loc_max_flux = loc_max_7(flux_of_3th, 1)
        
        """
        #beta=0.416767...あたりで使っていた条件
        for j in range(len(loc_max[0,:])):
            # if not  ((0.78 < loc_max[0,j] < 1.06) 
            #     #and (0.03 < loc_max[1,j] < 0.13 or 0.19 < loc_max[1,j] < 0.31)
            #     #and (0.12 < loc_max[2,j] < 0.26 or 0.38 < loc_max[2,j] < 0.46)
            #     #and (0.17 < loc_max[3,j] < 0.22 or 0.25 < loc_max[3,j] < 0.31 or 0.34 < loc_max[3,j] < 0.41)
            #     #and (0.09 < loc_max[4,j] < 0.195 or 0.285 < loc_max[4,j] < 0.34)
            #     #and (0.065 < loc_max[5,j] < 0.11 or 0.12 < loc_max[5,j] < 0.15)
            #     #and (0.035 < loc_max[6,j] < 0.07 or 0.105 < loc_max[6,j] < 0.125 or 0.135 < loc_max[6,j] < 0.17 or 0.18 < loc_max[6,j] < 0.21)
            #     #and (0.024 < loc_max[7,j] < 0.072 or 0.104 < loc_max[7,j] <0.136)
            #     #and (0.016 < loc_max[8,j] < 0.046 or 0.056 < loc_max[8,j] < 0.076)
            #     #and (0.004 < loc_max[9,j] < 0.032 or 0.05 < loc_max[9,j] < 0.072 or 0.086 < loc_max[9,j] < 0.1)
            #     #and (0 < loc_max[10,j] < 0.012 or 0.02 < loc_max[10,j] < 0.034 or 0.047 < loc_max[10,j] < 0.06)
            #     #and (0 < loc_max[11,j] < 0.0015 or 0.003 < loc_max[11,j] < 0.008 or 0.0135 < loc_max[11,j] < 0.0185)
            #     #and (0 < loc_max[12,j] < 0.0005 or 0.001 < loc_max[12,j] < 0.0018)
            #     #and (0 < loc_max[13,j] < 0.000005 or 0.00001 < loc_max[13,j] < 0.00003)
            #     ):
        """

        #単純に範囲を指定するとき
        try:
            if not (0.5 < np.min(np.abs(arr[0,:])) and np.max(np.abs(arr[0,:])) < 1.1 and 0.87 < np.min(loc_max_1st[0,:]) and np.max(loc_min_1st[0,:]) < 0.64):
                #print("ある")
                return zero #ダメだったらゼロ行列を返す(numbaを使う上でreturnの型を統一しなければならない)
            return arr[:,:int((progress_sec+1E-10)//step)+1].copy() #メモリを連続させる(C連続)ためにcopyにする #初期点も返す
        
        except:  #短すぎて極値がない時
            #極値どちらもない時
            if loc_max_1st.shape[1] == 0 and loc_min_1st.shape[1] == 0:
                #print("どっちもない")
                if not (0.5 < np.min(np.abs(arr[0,:])) and np.max(np.abs(arr[0,:])) < 1.1):
                    return zero
                return arr[:,:int((progress_sec+1E-10)//step)+1].copy()

            #極小値がない時
            elif loc_min_1st.shape[1] == 0:
                #print("小がない")
                if not (0.5 < np.min(np.abs(arr[0,:])) and np.max(np.abs(arr[0,:])) < 1.1 and 0.87 < np.min(loc_max_1st[0,:])):
                    return zero #ダメだったらゼロ行列を返す(numbaを使う上でreturnの型を統一しなければならない)
                return arr[:,:int((progress_sec+1E-10)//step)+1].copy()
            #極大値がない時
            elif loc_max_1st.shape[1] == 0:
                #print("大がない")
                if not (0.5 < np.min(np.abs(arr[0,:])) and np.max(np.abs(arr[0,:])) < 1.1 and np.max(loc_min_1st[0,:]) < 0.64):
                    return zero #ダメだったらゼロ行列を返す(numbaを使う上でreturnの型を統一しなければならない)
                return arr[:,:int((progress_sec+1E-10)//step)+1].copy()
            else:
                print("エラー")
        
        # #小ラミナー用の判定装置
        # if not (
        #     (0.48 < np.min(np.abs(arr[0, :])) and np.max(np.abs(arr[0, :])) < 0.55)
        #     and (0.18 < np.min(np.abs(arr[1, :])) and np.max(np.abs(arr[1, :])) < 0.27)
        #     and (0.19 < np.min(np.abs(arr[2, :])) and np.max(np.abs(arr[2, :])) < 0.33)
        #     and (0.1 < np.min(np.abs(arr[3, :])) and np.max(np.abs(arr[3, :])) < 0.3)
        #     and (0 < np.min(np.abs(arr[4, :])) and np.max(np.abs(arr[4, :])) < 0.28)
        #     and (0.06 < np.min(np.abs(arr[5, :])) and np.max(np.abs(arr[5, :])) < 0.18)
        #     and (0 < np.min(np.abs(arr[6, :])) and np.max(np.abs(arr[6, :])) < 0.2)
        #     and (0.01 < np.min(np.abs(arr[7, :])) and np.max(np.abs(arr[7, :])) < 0.14)
        #     and (0 < np.min(np.abs(arr[8, :])) and np.max(np.abs(arr[8, :])) < 0.066)
        #     and (0 < np.min(np.abs(arr[13, :])) and np.max(np.abs(arr[13, :])) < 0.000035)
        # ):
        #     return zero
        # return arr[:,:int((progress_sec+1E-10)//step)+1].copy()
        
    
    @staticmethod
    #@njit
    def get_laminar_wrapped(nu, f, k_n, c_n_1, c_n_2, c_n_3, start, step, period, rk4, function, loc_max_7, loc_min_7, judge_laminar_for_a_while, perturbator, flux, check_sec, progress_sec):
        dimention = len(start) #変数の次元
        step_number = abs(int((period[1] - period[0]+1E-10) // step)) #刻む回数
        
        arr = np.zeros((dimention+1,step_number+1),dtype = np.float64) #器を作る(1列のみ)
        arr = arr.astype(np.complex128) #複素数の時は実行
        arr[:-1,0:1] = start
        arr[-1,0] = period[0]
        i=0 #イテレーター
        perturbated_time = []
        cycle_limit = 5E+4
        while abs(arr[-1,-1]) == 0:
            if int(abs(arr[-1,i])+1E-08) % 10 == 0:
                print(f'{int(abs(arr[-1,i]))}時間', end = '') #時間ごとにプリント
            
            dummy = judge_laminar_for_a_while(nu, f, k_n, c_n_1, c_n_2, c_n_3, step, rk4, function, loc_max_7, loc_min_7, flux, check_sec, progress_sec, arr[:,i:i+1])
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
                    dummy = judge_laminar_for_a_while(nu, f, k_n, c_n_1, c_n_2, c_n_3, step, rk4, function, loc_max_7, loc_min_7, flux, check_sec, progress_sec, perturbated)
                    if n % 100 == 0:
                        if n == 100:
                            print(f'\n{n}試行目', end='')
                        print(f'\r {n}試行目', end='')
                    n += 1
                    if n > cycle_limit:
                        return arr[:,:i+1], perturbated_time #ダメだったらそれまでを返す
                    
                    if not dummy[0,0] == 0:
                        arr[:,i:i+len(dummy[0,:])] = dummy
                        i += len(dummy[0,:]) - 1 #イテレータを次の初期値(今回の最後)まで持ってく
                        if n > cycle_limit*0.8:
                            cycle_limit *=1.5
                        break
        return arr, perturbated_time
    def get_laminar(self, nu, f, k_n, c_n_1, c_n_2, c_n_3, start, step, period, check_sec, progress_sec):
        self.arr, self.perturbated_time = Long_laminar.get_laminar_wrapped(nu, f, k_n, c_n_1, c_n_2, c_n_3, start, step, period, Long_laminar.rk4, Long_laminar.goy_shell_model, Long_laminar.loc_max_7, Long_laminar.loc_min_7,Long_laminar.judge_laminar_for_a_while, Long_laminar.perturbator, Long_laminar.flux, check_sec, progress_sec)
        
    @staticmethod
    @njit(nogil=True, cache=True)
    def judge_laminar_or_not(laminar, orbit, skip, epsilon):
        """_summary_
            input orbit(not including time), and output whether each step is in laminar or not by binary array
        Args:
            laminar (_type_): laminar array
            orbit (_type_): orbit(not including time)
            skip (_type_): how often you judge the orbit
        """
        arr = np.zeros((1, orbit.shape[1]), dtype=np.int16)
        for i in range(round((orbit.shape[1]/skip))):
            a = np.abs(laminar[:4,:]) - np.abs(orbit[:4,skip*i:skip*i+1])
            a = a * a
            b = np.real(np.sum(a, axis = 0) ** (1/2))
            if b[np.where(b < epsilon)].shape[0] == 0:
                arr[:, skip*i:skip*(i+1)] = 0
            else:
                arr[:, skip*i:skip*(i+1)+1] = 1
        return arr