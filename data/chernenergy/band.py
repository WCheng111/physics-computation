import numpy as np
import matplotlib.pyplot as plt
import math
import time



M0=25
B3=-10
B=25
A=5
SOC=0
ky=0
def H(kx,ky,SOC):
    H=np.array([[M0+2*B3*(1-math.cos(kx))+2*B*(1-math.cos(ky)), -A*(math.sin(kx)-1j*math.sin(ky)), -A*(math.sin(kx)-1j*math.sin(ky))],
               [-A*(math.sin(kx)+1j*math.sin(ky)),-(M0+2*B3*(1-math.cos(kx))+2*B*(1-math.cos(ky))), 0],
                [-A*(math.sin(kx)-1j*math.sin(ky)), 0,-(M0+2*B3*(1-math.cos(kx))+2*B*(1-math.cos(ky)))+SOC]])
    return H

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
    sta0=sta[:,np.argsort(np.sort(eng))[0]]
    sta1=sta[:,np.argsort(np.sort(eng))[1]]
    sta2=sta[:,np.argsort(np.sort(eng))[2]]
    return eng0,eng1,eng2,sta0,sta1,sta2
# print(calate(0,0))
##write a function to calculate the derivative of the Hamiltonian
kx=np.linspace(-1,1,201)*math.pi
E=np.zeros((len(kx),3))
for i in range(len(kx)):
    E[i,:]=calate(math.pi+kx[i],ky,SOC)[0:3]
plt.plot(kx/math.pi,E[:,0],label='band1',color='black')
plt.plot(kx/math.pi,E[:,1],label='band2',color='black')
plt.plot(kx/math.pi,E[:,2],label='band3',color='black')
# plt.xlabel('kz')
plt.text(0,-33, r'$Z$',size=20)
plt.text(1,-33, r'$\Gamma$',size=20)
plt.text(-1,-33, r'$\Gamma$',size=20)
plt.axvline(0, color='k', linestyle='--',)
plt.axvline(-1, color='k', linestyle='--',)
plt.axvline(1, color='k', linestyle='--',)
plt.title(f'soc={SOC}')
plt.ylabel('Energy')
plt.xticks([])
plt.show()