import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import cmath
from numba import njit, f8,i8,c16
import numba
import matplotlib.animation as animation
import copy
import matplotlib

class Long_laminar:
    def __init__(self):
        self.arr = None
        self.arr = None
        self.average = None

    #微分方程式を定義
    @staticmethod
    @njit(cache=True)
    def henon(value, A = 3, B = 0.3, C = 5, D = 0.3, k = 0.4):
        """
        エノン写像
        
        """
        ddt_u = np.zeros((len(value[:,0]), 1), dtype = np.float64)
        """
        #微分方程式(4変数)
        ddt_u[0,:] = A - value[0,0]**2 + B * value[1,0] + k*(value[0,0] - value[2,0])
        ddt_u[1,:] = value[0,0]*1
        ddt_u[2,:] = C - value[2,0]**2 + D * value[3,0] + k * (value[2,0] - value[0,0])
        ddt_u[3,:] = value[2,0]*1
        """
        #微分方程式(2変数)
        a, b = 1.75, 0.3
        ddt_u[0,:] = 1 - a * value[0,0] ** 2 + value[1,0]
        ddt_u[1,:] = b * value[0,0]
        return ddt_u
    
    #摂動()を与える　*時間は入れない*
    @staticmethod
    @njit(cache=True)
    def perturbator(array):
        a = -13
        b = -15
        s = (b - a) * np.random.rand(len(array[:,0]),1) + a
        
        u = array * (2 * np.random.rand(len(array[:,0]),1) - 1)
        
        u_scaled = u / np.linalg.norm(u)
        
        return u_scaled * (10 ** s) * (-1) ** np.random.randint(0,2) + array


    @staticmethod
    #@njit(cache=True)
    def get_arr_wrapped(func, period, start):
        """ 
        ある時点から指定秒数走らせる
        """ 
        arr = np.zeros((len(start[:,0])+1, period[1]+1),dtype=np.float64) #入れ物を作る
        n = period[0]
        arr[:-1,0:1] = start
        for i in range(period[1]): #ここのfor文でルンゲクッタ実行
            arr[:-1,i+1:i+2]=func(arr[:-1,i:i+1])
            n += 1
            arr[-1,i+1] = n
        return arr
    
    def get_arr(self, period, start):
        return Long_laminar.get_arr_wrapped(Long_laminar.henon, period, start)
            
    @staticmethod
    @njit(cache=True)
    def judge_laminar_for_a_while(func, check, progress, start):
        """ 
        ある時点から指定秒数走らせて、ラミナーに入っているかどうか指定秒を返す
        *startにはtが入ってる*
        """ 
        arr = np.zeros((len(start), check+1),dtype=np.float64) #入れ物を作る
        n = start[-1,0]
        arr[:,0:1] = start
        for i in range(check): #ここのfor文でルンゲクッタ実行
            arr[:-1,i+1:i+2]=func(arr[:-1,i:i+1])
            n += 1
            arr[-1,i+1] = n
        

        if not (-4 < np.min(arr[:-1,:]) and np.max(arr[:-1,:])< 4):

            zero = np.zeros((len(arr[:,0]), int(progress)+1), dtype = np.float64)
            return zero #ダメだったらゼロ行列を返す(numbaを使う上でreturnの型を統一しなければならない)
        
        return arr[:,:progress+1] #初期点も返す
    
    @staticmethod
    #@njit
    def get_laminar_wrapped(start, period, function, judge_laminar_for_a_while, perturbator, check, progress):
        dimention = len(start) #変数の次元
        
        arr = np.zeros((dimention+1,period[1]+1),dtype = np.float64) #器を作る
        arr[:-1,0:1] = start
        arr[-1,0] = period[0]
        i=0 #イテレーター
        perturbated_time = []
        
        while abs(arr[-1,-1]) == 0:
            dummy = judge_laminar_for_a_while(function, check, progress, arr[:,i:i+1])
            if not dummy[0,0] == 0:
                arr[:,i:i+len(dummy[0,:])] = dummy
                i += len(dummy[0,:]) - 1 #イテレータを次の初期値(今回の最後)まで持ってく
                print(dummy[:-1,-1])
            else:
                perturbated_time.append(abs(arr[-1,i]))
                counter = 0 #摂動を加えた回数のカウンター
                while True:
                    #摂動を加えた点pertubatedを作る
                    perturbated = arr[:,i:i+1].copy() #最終地点をコピってくる
                    counter += 1
                    perturbated[:-1,:] = perturbator(perturbated[:-1,:]) #摂動を与える
                    dummy = judge_laminar_for_a_while(function, check, progress, arr[:,i:i+1])
                    if not dummy[0,0] == 0:
                        arr[:,i:i+len(dummy[0,:])] = dummy
                        i += len(dummy[0,:]) - 1 #イテレータを次の初期値(今回の最後)まで持ってく
                        print(dummy[:-1,-1],counter)
                        break
        return arr, perturbated_time
    def get_laminar(self, start, period, check, progress):
        self.arr, self.perturbated_time = Long_laminar.get_laminar_wrapped(start, period, Long_laminar.henon, Long_laminar.judge_laminar_for_a_while, Long_laminar.perturbator, check, progress)

def main():
    runge_para = {
    'start' :  np.array([[1.125442550244827], [0.481056895965219]]),
    'period' : np.array([0,50])
    }
    

    #刻み幅
    check = 40
    progress = 1

    lmodel = Long_laminar()
    lmodel.get_laminar(**runge_para, check = check, progress = progress)
main()