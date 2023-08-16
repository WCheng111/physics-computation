import numpy as np
import matplotlib.pyplot as plt

import math
import time


P=list(range(4))
P[0]=np.array([[1, 0], [0, 1]])
P[1]=np.array([[0, 1], [1, 0]])
P[2]=np.array([[0, -1j], [1j, 0]])
P[3]=np.array([[1, 0], [0, -1]])

A=0.5
B=0.5
C=0.5
D=0.5





def H(kx,ky,M):
    H=(C-2*D*(2-math.cos(kx)-math.cos(ky)))*P[0]+(M-4*B+2*B*(math.cos(kx)+math.cos(ky)))*P[3]+A*math.sin(kx)*P[1]+A*math.sin(ky)*P[2]
    return H
#print(H(0,0))
def cal1(ks1,M):
    Eng=[]
    for i in range(len(ks1)):
        kx=ks1[i]
        ky=0
        eng, sta=np.linalg.eigh(H(kx,ky,M))
        Eng.append(eng)
    return Eng
#write a function to calculate the eigenvalues and eigenstates of the Hamiltonian
##返回中间band的能量和波函数
def calate(kx,ky,M):
    eng, sta=np.linalg.eigh(H(kx,ky,M))
    eng0=np.sort(eng)[0]
    eng1=np.sort(eng)[1]
    sta0=sta[:,np.argsort(np.sort(eng))[0]]
    sta1=sta[:,np.argsort(np.sort(eng))[1]]
    return eng0,eng1,sta0,sta1
# print(calate(0,0,0))
##write a function to calculate the derivative of the Hamiltonian
def diffx(kx,ky,M):
    diffx=-2*D*math.sin(kx)*P[0]-2*B*math.sin(kx)*P[3]+A*math.cos(kx)*P[1]
    return diffx
def diffy(kx,ky,M):
    delta=1e-15
    diffy=-2*D*math.sin(ky)*P[0]-2*B*math.sin(ky)*P[3]+A*math.cos(ky)*P[2]
    return diffy
# print(diffx(0,0))
#print(np.dot(sta0,np.dot(diffx(0,0),sta0)))
# print(np.conjugate(sta0))
N=300
kx=np.linspace(-math.pi,math.pi,N)
ky=np.linspace(-math.pi,math.pi,N)
Delta=2*math.pi/N
chern_1=0
chern_2=0
chern_3=0

# M=1.5
#
# for i in range(len(kx)-1):
#      for j in range(len(ky)-1):
#         eng0,eng1,sta0,sta1=calate(kx[i],ky[j],M)
#         partial01x=np.dot(sta0.transpose().conj(),np.dot(diffx(kx[i]+Delta/2,ky[j]+Delta/2,M),sta1))*np.dot(sta1.transpose().conj(),np.dot(diffy(kx[i]+Delta/2,ky[j]+Delta/2,M),sta0))
#         partial01y=np.dot(sta0.transpose().conj(),np.dot(diffy(kx[i]+Delta/2,ky[j]+Delta/2,M),sta1))*np.dot(sta1.transpose().conj(),np.dot(diffx(kx[i]+Delta/2,ky[j]+Delta/2,M),sta0))
#         chern_1=chern_1+((partial01x-partial01y)/((eng0-eng1)**2))*Delta**2
#
# for i in range(len(kx) - 1):
#     for j in range(len(ky) - 1):
#         eng0, eng1, sta0, sta1 = calate(kx[i], ky[j],M)
#         partial10x = np.dot(sta1.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,M), sta0)) * np.dot(sta0.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,M), sta1))
#         partial10y = np.dot(sta1.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,M), sta0)) * np.dot(sta0.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,M), sta1))
#         chern_2 = chern_2 + ((partial10x - partial10y) / ((eng1 - eng0) ** 2))*Delta**2


Mchange=np.linspace(-5.2,6.2,5)
print(Mchange)
chern_change=np.zeros((len(Mchange),2))
for k in range(len(Mchange)):
    chern_change1 = 0
    chern_change2 = 0
    for i in range(len(kx) - 1):
        for j in range(len(ky) - 1):
            eng0, eng1, sta0, sta1 = calate(kx[i]+Delta/2, ky[j]+Delta/2, Mchange[k])
            partial01x = np.dot(sta0.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,Mchange[k]), sta1)) * np.dot(
                sta1.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,Mchange[k]), sta0))
            partial01y = np.dot(sta0.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,Mchange[k]), sta1)) * np.dot(
                sta1.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,Mchange[k]), sta0))
            chern_change1 = chern_change1 + ((partial01x - partial01y) / ((eng0 - eng1) ** 2)) * Delta ** 2
    for i in range(len(kx) - 1):
        for j in range(len(ky) - 1):
            eng0, eng1, sta0, sta1 = calate(kx[i]+Delta/2, ky[j]+Delta/2,Mchange[k])
            partial10x = np.dot(sta1.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,Mchange[k]), sta0)) * np.dot(
                sta0.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,Mchange[k]), sta1))
            partial10y = np.dot(sta1.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,Mchange[k]), sta0)) * np.dot(
                sta0.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,Mchange[k]), sta1))
            chern_change2 = chern_change2 + ((partial10x - partial10y) / ((eng1 - eng0) ** 2)) * Delta ** 2
    chern_change[k,0]=(1j*chern_change1/(2*math.pi)).real
    chern_change[k,1]=(1j*chern_change2/(2*math.pi)).real

#
plt.scatter(Mchange,chern_change[:,0])
plt.plot(Mchange,chern_change[:,1])
plt.xlim(-5,5)
plt.xlabel("M")
plt.ylabel("Chern number")
plt.show(block=True)
# print("下面能带的陈数为:",(1j*chern_1)/(2*math.pi))
# print("上面能带的陈数为:",(1j*chern_2)/(2*math.pi))


# A=np.zeros((3,2))
# print(A)
# Eng=cal1(kx,0.5)
# plt.plot(kx,Eng)
# plt.show()



