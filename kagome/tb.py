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


d=0.045
p=0.9085
t1=0.955
t2=-0.41
t11=0.0878
t22=-0.072
t3=-0.02
tso=0.09
c=1
a=0.1



def cal_energy(ksx,ksy,ksz):
    kx1=ksx
    ky1=ksy
    kz1=ksz
    def Hd(kx,ky,kz):
        Hd=np.array([[d+2*t22*cos(c*kz) ,t2*(1+exp(1j*(-a/2*kx+sqrt(3)/2*a*ky))) ,t2*(1+exp(1j*a*kx)) ],
                  [t2*(1+exp(-1j*c*kz))  , d+2*t22*cos(c*kz) ,  t2*(exp(1j*a*kx)+exp(-1j*(-a/2*kx+sqrt(3)/2*a*ky)))],
                  [t2*(1+exp(-1j*a*kx)) ,t2*(exp(-1j*a*kx)+exp(1j*(-a/2*kx+sqrt(3)/2*a*ky))) ,d+2*t22*cos(c*kz)],
                  [-2j*t3*sin(c*kz) , -2j*t3*sin(c*kz)*exp(1j*(-a/2*kx+sqrt(3)/2*a*ky)), -2j*t3*sin(c*kz)],
                  [-2j*t3*sin(c*kz) ,-2j*t3*sin(c*kz), -2j*t3*sin(c*kz)*exp(1j*a*kz)]])
        return Hd
    def Hp(kx,ky,kz):
        Hp=tso*np.array([[0 ,0, 0],
                  [0 ,0 ,0 ],
                  [0 ,0 ,0 ],
                  [exp(1j*5*pi/6), exp(-1j*pi/2)*exp(1j*(-a/2*kx+sqrt(3)/2*a*ky)), exp(1j*pi/6)],
                  [exp(-1j*pi/6), exp(1j*pi/2) ,exp(-1j*5*pi/6)*exp(1j*a*kx)]])
        return Hp
    H=np.bmat([[Hd(kx1,ky1,kz1),-np.conjugate(Hp(-kx1,-ky1,-kz1))],[Hp(kx1,ky1,kz1),np.conjugate(Hd(-kx1,-ky1,-kz1))]])
    #E=sla.eigsh(H, k=1, which='SA', return_eigenvectors=False)
    #print(H)
    eng, sta = np.linalg.eigh(H)
    return eng

Eng=[]
Eng1=[]
gammam=np.linspace(0,2/(3*a),200)*pi
for i in range(len(gammam)):
    Eng.append(cal_energy(gammam[i],0,0))
    Eng1.append(cal_energy(gammam[i],0,0))
mk=np.linspace(0,2/(sqrt(3)*3*a),200)*pi
mkx=np.linspace(2/(3*a),2/(3*a)+2/(sqrt(3)*3*a),200)*pi
for i in range(len(mk)):
    Eng.append(cal_energy(2/(3*a)*pi,mk[i],0))
kgamma=np.linspace(4/(3*sqrt(3)*a),0,200)*pi
kgammax=np.linspace(2/(3*a)+2/(sqrt(3)*3*a),2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3)),200)*pi
for i in range(len(kgamma)):
    Eng.append(cal_energy(kgamma[i]*sqrt(3)/2,kgamma[i]/2,0))
kx=[]
#print(len(gammam+mk))
#plt.plot(kx,Eng)
for i in range(len(gammam)):
    kx.append(gammam[i])
for i in range(len(mkx)):
    kx.append(mkx[i])
for i in range(len(kgammax)):
    kx.append(kgammax[i])

#gamma a
gammaA=np.linspace(0,1/c,200)*pi
gammaAx=np.linspace(2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3)),1/c+2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3)),200)*pi

for i in range(len(gammaA)):
    Eng.append(cal_energy(0,0,gammaA[i]))
for i in range(len(gammaAx)):
    kx.append(gammaAx[i])

AL=np.linspace(0,2/(3*a),200)*pi
ALX=np.linspace(1/c+2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3)),1/c+2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3))+2/(3*a),200)*pi

for i in range(len(AL)):
    Eng.append(cal_energy(AL[i],0,pi/c))
for i in range(len(ALX)):
    kx.append(ALX[i])

LH=np.linspace(0,2/(sqrt(3)*3*a),200)*pi
LHx=np.linspace(1/c+2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3))+2/(3*a),1/c+2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3))+2/(3*a)+2/(sqrt(3)*3*a),200)*pi

for i in range(len(LH)):
    Eng.append(cal_energy(2/(3*a)*pi,LH[i],pi/c))
for i in range(len(LHx)):
    kx.append(LHx[i])

HA=np.linspace(4/(3*a*sqrt(3)),0,200)*pi
HAx=np.linspace(1/c+2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3))+2/(3*a)+2/(sqrt(3)*3*a),1/c+2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3))+2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3)),200)*pi

for i in  range(len(HA)):
    Eng.append(cal_energy(HA[i]*sqrt(3)/2,HA[i]/2,pi/c))
for i in range(len(HAx)):
    kx.append(HAx[i])
#plt.plot(gammam,Eng1)
plt.plot(kx,Eng)
#plt.ylim(-1,1)
plt.axvline(0, color='k', linestyle='--',)
plt.axvline(2*pi/(3*a), color='k', linestyle='--',)
plt.axvline((2/(3*a)+2/(sqrt(3)*3*a))*pi, color='k', linestyle='--',)
plt.axvline((2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3)))*pi, color='k', linestyle='--',)
plt.axvline((1/c+2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3)))*pi, color='k', linestyle='--',)
plt.axvline((1/c+2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3))+2/(3*a))*pi, color='k', linestyle='--',)
plt.axvline((1/c+2/(3*a)+2/(sqrt(3)*3*a)+4/(3*a*sqrt(3))+2/(3*a)+2/(sqrt(3)*3*a))*pi, color='k', linestyle='--',)

plt.text(-2,0, r'$\Gamma$')
plt.show()
print(gammam)
print(kx)
