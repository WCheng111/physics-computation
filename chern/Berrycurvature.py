import numpy as np
import matplotlib.pyplot as plt
import math
import time
from matplotlib.colors import Normalize


M0=25
M1=-25
A=12.5


def H(kx,ky,SOC):
    H=np.array([[M0+2*M1*(1-math.cos(kx))+2*M1*(1-math.cos(ky)), -1j*A*(math.sin(kx)-1j*math.sin(ky)), -1j*A*(math.sin(kx)-1j*math.sin(ky))],
               [1j*A*(math.sin(kx)+1j*math.sin(ky)),-(M0+2*M1*(1-math.cos(kx))+2*M1*(1-math.cos(ky))), 0],
                [1j*A*(math.sin(kx)+1j*math.sin(ky)), 0,-(M0+2*M1*(1-math.cos(kx))+2*M1*(1-math.cos(ky)))+SOC]])
    return H



#print(H(0,0))
# def cal1(ks1,SOC):
#     Eng=[]
#     for i in range(len(ks1)):
#         kx=ks1[i]
#         ky=0
#         eng, sta=np.linalg.eigh(H(kx,ky,SOC))
#         Eng.append(eng)
#     return Eng
#write a function to calculate the eigenvalues and eigenstates of the Hamiltonian
##返回中间band的能量和波函数
def calate(kx,ky,SOC):
    eng, sta=np.linalg.eigh(H(kx,ky,SOC))
    eng0=np.sort(eng)[0]
    eng1=np.sort(eng)[1]
    eng2=np.sort(eng)[2]
    sta0=sta[:,np.argsort(eng)[0]]
    sta1=sta[:,np.argsort(eng)[1]]
    sta2=sta[:,np.argsort(eng)[2]]
    return eng0,eng1,eng2,sta0,sta1,sta2
# print(calate(0,0))
##write a function to calculate the derivative of the Hamiltonian
def diffx(kx,ky,SOC):
    diffx=np.array([[2*M1*math.sin(kx), -1j*A*math.cos(kx), -1j*A*math.cos(kx)],
               [1j*A*math.cos(kx),-2*M1*math.sin(kx), 0],
                [1j*A*math.cos(kx), 0,-2*M1*math.sin(kx)+SOC]])
    return diffx
def diffy(kx,ky,SOC):
    diffy=np.array([[2*M1*math.sin(ky), -1*A*math.cos(ky),-A*math.cos(ky)],
               [-1*A*math.cos(ky),-2*M1*math.sin(ky), 0],
                [-A*math.cos(ky), 0,-2*M1*math.sin(ky)+SOC]])
    return diffy
# print(diffx(0,0))
#print(np.dot(sta0,np.dot(diffx(0,0),sta0)))
# print(np.conjugate(sta0))
N=200
kx=np.linspace(-math.pi,math.pi,N)
ky=np.linspace(-math.pi,math.pi,N)
Delta=2*math.pi/N
# chern_1=0
# chern_2=0
# chern_3=0
SOCl=[10,20,30,40]
# chern_change=np.zeros((len(SOC),3))
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs = axs.ravel()
for k, SOC in enumerate(SOCl):
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
# Berry_min=Berry.min()
# Berry_max=Berry.max()
# plt.imshow(Berry,extent=[-math.pi,math.pi,-math.pi,math.pi],cmap='coolwarm',origin='lower',norm=Normalize(vmin=Berry_min,vmax=Berry_max))
# plt.colorbar()
# plt.show(block=True)

# np.save('chernnum.npy',chern_change)
# plt.scatter(SOC, chern_change[:, 0], label='chern_1')
# plt.scatter(SOC, chern_change[:, 1], label='chern_2')
# plt.scatter(SOC, chern_change[:, 2], label='chern_3')
# plt.show()
# end=time.time print(end-start)
# print("最下面能带的陈数为",1j*chern_1/(2*math.pi))()
# #
# print("中间能带的陈数为",1j*chern_2/(2*math.pi))
# print("最上面能带的陈数为",1j*chern_3/(2*math.pi))
# ks=np.linspace(-math.pi,math.pi,100)
# Eng1=cal1(ks)

# A=np.array([[1,2,3],[2,3,3],[3,4,3]])
# B=np.array([1j,2*1j,3*1j])
# C=np.array([1,2,3])
# print(np.dot(A,B))
# print(np.dot(np.dot(B.conj().transpose(),A),C))
# print(2**2)
# print(kx)
#plt.plot(ks,Eng1)
#plt.show()