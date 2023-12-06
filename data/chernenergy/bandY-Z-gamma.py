import numpy as np
import matplotlib.pyplot as plt
import math
import time



M0=25
B3=-10
B=25
A=5
SOC=30
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
Zgamma=np.linspace(0,1,201)*math.pi
Yz=np.linspace(-1,0,201)*math.pi
kx=np.linspace(-1,1,2*len(Zgamma))
E=np.zeros((2*len(Zgamma),3))
for i in range(len(Yz)):
    E[i,:]=calate(math.pi,Yz[i],SOC)[0:3]
for j in range(len(Zgamma)):
    E[len(Yz)+j, :] = calate(math.pi-Zgamma[j], 0, SOC)[0:3]

plt.plot(kx,E[:,0],label='band1',color='black')
plt.plot(kx,E[:,1],label='band2',color='black')
plt.plot(kx,E[:,2],label='band3',color='black')
plt.axvline(x=0)
plt.axvline()
plt.xlabel('kz')
plt.title(f'soc={SOC}')
plt.ylabel('Energy')
plt.show()