import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cmath

class RungeKutta:
    def __init__(self,function):
        self.f = function
        self.arr = None
        
    def rk4(self,x,y):
        #xは時間、yは前の値。1
        k1 = self.step * self.f(x, y, **self.parameter)
        k2 = self.step * self.f(x + self.step / 2, y + k1 / 2, **self.parameter)
        k3 = self.step * self.f(x + self.step / 2, y + k2 / 2, **self.parameter)
        k4 = self.step * self.f(x + self.step / 2, y + k3, **self.parameter)
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def get_arr(self,parameter,start,step=0.1,period=(0,100),progress_bar = False,complex = False):
        if isinstance(start,np.ndarray):
            dimention = len(start)
        else:
            dimention = 1
        
        self.step = step
        self.parameter = parameter
        step_number = abs(int((period[1] - period[0]) // self.step)) #刻む回数
        self.arr = np.zeros((dimention+1,step_number+1),dtype = np.float64)
        
        if complex is True:
            self.arr = self.arr.astype(np.complex128)
        
        t = period[0]
        self.arr[:-1,0:] = start
        self.arr[-1,0:] = t
        
        if progress_bar is False:
            for i in range(step_number): #ここのfor文でルンゲクッタ実行
                self.arr[:-1,i+1:i+2]=self.rk4(t,self.arr[:-1,i:i+1])
                t+=self.step
                self.arr[-1,i+1]=t
        
        else:
            for i in tqdm(range(step_number)): #ここのfor文でルンゲクッタ実行
                self.arr[:-1,i+1]=self.rk4(t,self.arr[:-1,i])
                t+=self.step
                self.arr[-1,i+1]=t
                
        #もしこのクラスでplotがしたいならget.arrの引数にplot(bool型),label=("x","y","z")を足して
#         #以下プロット
#         if plot is True:
#             if dimention==3:
#                 x=self.arr[0, :]
#                 y=self.arr[1, :]
#                 z=self.arr[2, :]
                
#                 fig=plt.figure()
#                 ax = fig.add_subplot(projection='3d')
#                 plt.plot(x,y,z,alpha=0.9)
#                 ax.set_xlabel(label[0])
#                 ax.set_ylabel(label[1])
#                 ax.set_zlabel(label[2])
#                 plt.show()
            
#             if dimention ==1:
#                 x=self.arr[1,:]
#                 y=self.arr[0,:]
                
#                 plt.figure()
#                 plt.plot(x,y)
#                 plt.xlabel(label[0])
#                 plt.ylabel(label[1])