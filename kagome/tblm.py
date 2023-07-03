import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import kwant
from matplotlib import pyplot as plt
import tinyarray
import numpy as np
from numpy import sqrt
import time
import scipy.sparse.linalg as sla
from numpy import exp, pi, cos, sin, arccos, arcsin, sign, kron
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from pathos.multiprocessing import Pool
import math
from mpl_toolkits import mplot3d


d=0
p=1
t1=1
t2=-0.414
t11=0.098
t22=-0.08
t3=-0.02
tso=0.09
c=1
a=0.1



def cal_energy(ksx,ksy,ksz):
    kx1=ksx
    ky1=ksy
    kz1=ksz
    def Hd(kx,ky,kz):
        Hd=np.array([[d+2*t22*cos(c*kz) ,t2*(1+exp(1j*(-a/2*kx+sqrt(3)/2*a*ky))) ,t2*(1+exp(1j*a*kx)) ,2*1j*t3*sin(c*kz), 2*1j*t3*sin(c*kz)],
                  [t2*(1+exp(-1j*(-a/2*kx+sqrt(3)/2*a*ky))) , d+2*t22*cos(c*kz) ,  t2*(exp(1j*a*kx)+exp(-1j*(-a/2*kx+sqrt(3)/2*a*ky)))  ,2j*t3*sin(c*kz)*exp(-1j*(-a/2*kx+sqrt(3)/2*a*ky)) , 2j*t3*sin(c*kz)],
                  [t2*(1+exp(-1j*a*kx)) ,t2*(exp(-1j*a*kx)+exp(1j*(-a/2*kx+sqrt(3)/2*a*ky))) ,d+2*t22*cos(c*kz) ,2j*t3*sin(c*kz) , 2j*t3*sin(c*kz)*exp(-1j*a*kx)],
                  [-2j*t3*sin(c*kz) , -2j*t3*sin(c*kz)*exp(1j*(-a/2*kx+sqrt(3)/2*a*ky)), -2j*t3*sin(c*kz) ,p+2*t11*cos(c*kz), t1*(1+exp(1j*(-a/2*kx+sqrt(3)/2*a*ky))+exp(-1j*a*kx))],
                  [-2j*t3*sin(c*kz) ,-2j*t3*sin(c*kz), -2j*t3*sin(c*kz)*exp(1j*a*kz) ,t1*(1+exp(-1j*(-a/2*kx+sqrt(3)/2*a*ky))+exp(1j*a*kx)), p+2*t11*cos(c*kz)]])
        return Hd
    def Hp(kx,ky,kz):
        Hp=tso*np.array([[0 ,0, 0 ,exp(-1j*pi/6) ,exp(1j*5*pi/6)],
                  [0 ,0 ,0 ,exp(1j*pi/2)*exp(-1j*(-a/2*kx+sqrt(3)/2*a*ky)) ,exp(-1j*pi/2)],
                  [0 ,0 ,0 ,exp(-1j*5*pi/6) ,exp(1j*pi/6)*exp(-1j*a*kx)],
                  [exp(1j*5*pi/6), exp(-1j*pi/2)*exp(1j*(-a/2*kx+sqrt(3)/2*a*ky)), exp(1j*pi/6) ,0 ,0],
                  [exp(-1j*pi/6), exp(1j*pi/2) ,exp(-1j*5*pi/6)*exp(1j*a*kx), 0, 0]])
        return Hp
    H=np.bmat([[Hd(kx1,ky1,kz1),-np.conjugate(Hp(-kx1,-ky1,-kz1))],[Hp(kx1,ky1,kz1),np.conjugate(Hd(-kx1,-ky1,-kz1))]])
    #E=sla.eigsh(H, k=1, which='SA', return_eigenvectors=False)
    #print(H)
    eng, sta = np.linalg.eigh(H)
    return eng

Eng=[]
Eng1=[]
LM=np.linspace(0,1/c,2)*pi
for i in range(len(LM)):
    Eng.append(cal_energy(0,2*pi/(sqrt(3)*a),LM[i]))

plt.plot(LM,Eng)
#plt.text(-2,0, r'$\Gamma$')
plt.hlines(0,0,3,colors='r', linestyles='--')
plt.show()

