import numpy as np
import matplotlib.pyplot as plt
import math
import time
from matplotlib.colors import Normalize

M0=25
B3=-10
B=25
A=2
SOC=30

def H(kx,ky,A3):
    H=np.array([[M0+2*B3*(1-math.cos(kx))+2*B*(1-math.cos(ky)), -A*(math.sin(kx)-1j*math.sin(ky)), -A3*math.sin(kx)+1j*A*math.sin(ky)],
               [-A*(math.sin(kx)+1j*math.sin(ky)),-(M0+2*B3*(1-math.cos(kx))+2*B*(1-math.cos(ky))), 0],
                [-A3*math.sin(kx)-1j*A*math.sin(ky), 0,-(M0+2*B3*(1-math.cos(kx))+2*B*(1-math.cos(ky)))+SOC]])
    return H

def calate(kx,ky,A3):
    eng, sta=np.linalg.eigh(H(kx,ky,A3))
    eng0=np.sort(eng)[0]
    eng1=np.sort(eng)[1]
    eng2=np.sort(eng)[2]
    sta0=sta[:,np.argsort(np.sort(eng))[0]]
    sta1=sta[:,np.argsort(np.sort(eng))[1]]
    sta2=sta[:,np.argsort(np.sort(eng))[2]]
    return eng0,eng1,eng2,sta0,sta1,sta2
# print(calate(0,0))
##write a function to calculate the derivative of the Hamiltonian
def diffx(kx,ky,A3):
    diffx=np.array([[2*B3*math.sin(kx), -A*math.cos(kx), -A3*math.cos(kx)],
               [-A*math.cos(kx),-2*B3*math.sin(kx), 0],
                [-A3*math.cos(kx), 0,-2*B3*math.sin(kx)]])
    return diffx
def diffy(kx,ky,A3):
    diffy=np.array([[2*B*math.sin(ky), A*1j*math.cos(ky), A*1j*math.cos(ky)],
               [-1j*A*math.cos(ky),-2*B*math.sin(ky), 0],
                [-1j*A*math.cos(ky), 0,-2*B*math.sin(ky)]])
    return diffy

N=200
kx=np.linspace(-math.pi,math.pi,N)
ky=np.linspace(-math.pi,math.pi,N)
Delta=2*math.pi/N
# chern_1=0
# chern_2=0
# chern_3=0
SOCl=[1,20,40,60]
# chern_change=np.zeros((len(SOC),3))
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()
Berry=np.zeros((len(kx),len(ky)))
for i in range(len(kx)):
     for j in range(len(ky)):
        eng0,eng1,eng2,sta0,sta1,sta2=calate(kx[i],ky[j],SOC)
        partial01x=np.dot(sta1.transpose().conj(),np.dot(diffx(kx[i],ky[j],SOC),sta0))*np.dot(sta0.transpose().conj(),np.dot(diffy(kx[i],ky[j],SOC),sta1))
        partial01y=np.dot(sta1.transpose().conj(),np.dot(diffy(kx[i],ky[j],SOC),sta0))*np.dot(sta0.transpose().conj(),np.dot(diffx(kx[i],ky[j],SOC),sta1))
        partial02x=np.dot(sta1.transpose().conj(),np.dot(diffx(kx[i],ky[j],SOC),sta2))*np.dot(sta2.transpose().conj(),np.dot(diffy(kx[i],ky[j],SOC),sta1))
        partial02y=np.dot(sta1.transpose().conj(),np.dot(diffy(kx[i],ky[j],SOC),sta2))*np.dot(sta2.transpose().conj(),np.dot(diffx(kx[i],ky[j],SOC),sta1))
        Berry[i,j]=(1j*((partial01x-partial01y)/((eng1-eng0)**2)+(partial02x-partial02y)/((eng1-eng2)**2))).real
im = axs[k].imshow(Berry, extent=(-math.pi, math.pi, -math.pi, math.pi),cmap='coolwarm',vmin=-10,vmax=10)
axs[k].set_title(f'SOC = {SOC}meV')
fig.colorbar(im, ax=axs[k])
plt.tight_layout()
plt.show()