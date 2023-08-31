import numpy as np
import matplotlib.pyplot as plt
import math
import time



M0=25
M1=-25
A=12.5
SOC=-80
ky=0
def H(kx,ky,SOC):
    H=np.array([[M0+2*M1*(1-math.cos(kx))+2*M1*(1-math.cos(ky)), -1j*A*(math.sin(kx)-1j*math.sin(ky)), -1j*A*(math.sin(kx)+1j*math.sin(ky))],
               [1j*A*(math.sin(kx)+1j*math.sin(ky)),-(M0+2*M1*(1-math.cos(kx))+2*M1*(1-math.cos(ky))), 0],
                [1j*A*(math.sin(kx)-1j*math.sin(ky)), 0,-(M0+2*M1*(1-math.cos(kx))+2*M1*(1-math.cos(ky)))+SOC]])
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
    sta0=sta[:,np.argsort(np.sort(eng))[0]]
    sta1=sta[:,np.argsort(np.sort(eng))[1]]
    sta2=sta[:,np.argsort(np.sort(eng))[2]]
    return eng0,eng1,eng2,sta0,sta1,sta2
# print(calate(0,0))
##write a function to calculate the derivative of the Hamiltonian
kx=np.linspace(-1,1,201)*math.pi
E=np.zeros((len(kx),3))
for i in range(len(kx)):
    E[i,:]=calate(kx[i],ky,SOC)[0:3]
plt.plot(kx,E[:,0],label='band1',color='black')
plt.plot(kx,E[:,1],label='band2',color='black')
plt.plot(kx,E[:,2],label='band3',color='black')
plt.xlabel('kx')
plt.title(f'soc={SOC}')
plt.ylabel('Energy')
plt.show()