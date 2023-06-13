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
c=0.9
a=1



def cal_energy(ksx,ksy,ksz):
    kx1=ksx
    ky1=ksy
    kz1=ksz
    def Hd(kx,ky,kz):
        Hd=np.array([[d+2*t22*cos(c*kz) ,t2*(1+exp(1j*(-a/2*kx+sqrt(3)/2*a*ky))) ,t2*(1+exp(1j*a*kx)) ,2*1j*t3*sin(c*kz), 2*1j*t3*sin(c*kz)],
                  [t2*(1+exp(-1j*(-a/2*kx+sqrt(3)/2*a*ky)))  , d+2*t22*cos(c*kz) ,  t2*(exp(1j*a*kx)+exp(-1j*(-a/2*kx+sqrt(3)/2*a*ky)))  ,2j*t3*sin(c*kz)*exp(-1j*(-a/2*kx+sqrt(3)/2*a*ky)) , 2j*t3*sin(c*kz)],
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
gammam=np.linspace(0,2/(sqrt(3)*a),200)*pi
for i in range(len(gammam)):
    Eng.append(cal_energy(0,gammam[i],0))
    Eng1.append(cal_energy(gammam[i],0,0))
mk=np.linspace(0,2/(3*a),200)*pi
mkx=np.linspace(2/(sqrt(3)*a),2/(sqrt(3)*a)+2/(3*a),200)*pi
for i in range(len(mk)):
    Eng.append(cal_energy(-mk[i],2*pi/(sqrt(3)*a),0))
kgamma=np.linspace(2/(sqrt(3)*a),0,200)*pi
kgammax=np.linspace(2/(sqrt(3)*a)+2/(3*a),2/(sqrt(3)*a)+2/(3*a)+2/(sqrt(3)*a),200)*pi
for i in range(len(kgamma)):
     Eng.append(cal_energy(-kgamma[i]/sqrt(3),kgamma[i],0))
gammaA = np.linspace(0, 1 / c, 200) * pi
gammaAx=np.linspace(2/(sqrt(3)*a)+2/(3*a)+2/(sqrt(3)*a),1/c+2/(sqrt(3)*a)+2/(3*a)+2/(sqrt(3)*a),200)*pi
for i in range(len(gammaA)):
    Eng.append(cal_energy(0,0,gammaA[i]))
kx=[]
AL=np.linspace(0,2/(sqrt(3)*a),200)*pi
ALx=np.linspace(1/c+2/(sqrt(3)*a)+2/(3*a)+2/(sqrt(3)*a),1/c+2/(sqrt(3)*a)+2/(3*a)+2/(sqrt(3)*a)+2/(sqrt(3)*a),200)*pi
for i in range(len(AL)):
    Eng.append(cal_energy(0,AL[i],pi/c))
#print(len(gammam+mk))
#plt.plot(kx,Eng)
LH=np.linspace(0,2/(3*a),200)*pi
LHx=np.linspace(1/c+2/(sqrt(3)*a)+2/(3*a)+2/(sqrt(3)*a)+2/(sqrt(3)*a),1/c+2/(sqrt(3)*a)+2/(3*a)+2/(sqrt(3)*a)+2/(sqrt(3)*a)+2/(3*a),200)*pi
for i in  range(len(LH)):
    Eng.append(cal_energy(-LH[i],2*pi/(sqrt(3)*a),pi/c))
HA=np.linspace(2/(sqrt(3)*a),0,200)*pi
HAx=np.linspace(1/c+2/(sqrt(3)*a)+2/(3*a)+2/(sqrt(3)*a)+2/(sqrt(3)*a)+2/(3*a),1/c+2/(sqrt(3)*a)+2/(3*a)+2/(sqrt(3)*a)+2/(sqrt(3)*a)+2/(3*a)+2/(sqrt(3)*a),200)*pi
for i in range(len(HA)):
    Eng.append(cal_energy(-HA[i]/sqrt(3),HA[i],pi/c))

for i in range(len(gammam)):
    kx.append(gammam[i])
for i in range(len(mkx)):
     kx.append(mkx[i])
for i in range(len(kgammax)):
     kx.append(kgammax[i])
for i in range(len(gammaAx)):
     kx.append(gammaAx[i])
for i in range(len(ALx)):
        kx.append(ALx[i])
for i in range(len(LHx)):
        kx.append(LHx[i])
for i in range(len(HAx)):
        kx.append(HAx[i])
plt.plot(kx,Eng)


plt.axvline(2*pi/(sqrt(3)*a), color='k', linestyle='--')
plt.annotate(r'$\Gamma$', xy=(0, 0), xytext=(0.5, 0.5))
plt.axvline(2*pi/(sqrt(3)*a)+2*pi/(3*a), color='k', linestyle='--')
plt.axvline(2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a), color='k', linestyle='--')
plt.axvline(pi/c+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a), color='k', linestyle='--')
plt.axvline(pi/c+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a)+2*pi/(sqrt(3)*a), color='k', linestyle='--')
plt.axvline(pi/c+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a)+2*pi/(sqrt(3)*a)+2*pi/(3*a), color='k', linestyle='--')
plt.axvline(pi/c+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a)+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a), color='k', linestyle='--')
plt.hlines(0,0,22,colors='r', linestyles='--')

plt.annotate(r'$\Gamma$', xy=(0, 0), xytext=(0, -2.5),color='r',fontsize=15)
plt.annotate(r'$M$', xy=(2*pi/(sqrt(3)*a), 0), xytext=(2*pi/(sqrt(3)*a), -2.5),color='r',fontsize=15)
plt.annotate(r'$K$', xy=(2*pi/(sqrt(3)*a)+2*pi/(3*a), 0), xytext=(2*pi/(sqrt(3)*a)+2*pi/(3*a), -2.5),color='r',fontsize=15)
plt.annotate(r'$\Gamma$', xy=(2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a), 0), xytext=(2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a), -2.5),color='r',fontsize=15)
plt.annotate(r'$A$', xy=(pi/c+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a), 0), xytext=(pi/c+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a), -2.5),color='r',fontsize=15)
plt.annotate(r'$L$', xy=(pi/c+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a)+2*pi/(sqrt(3)*a), 0), xytext=(pi/c+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a)+2*pi/(sqrt(3)*a), -2.5),color='r',fontsize=15)
plt.annotate(r'$H$', xy=(pi/c+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a)+2*pi/(sqrt(3)*a)+2*pi/(3*a), 0), xytext=(pi/c+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a)+2*pi/(sqrt(3)*a)+2*pi/(3*a), -2.5),color='r',fontsize=15)
plt.annotate(r'$A$', xy=(pi/c+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a)+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a), 0), xytext=(pi/c+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a)+2*pi/(sqrt(3)*a)+2*pi/(3*a)+2*pi/(sqrt(3)*a), -2.5),color='r',fontsize=15)
x_line=1

plt.show()