import numpy as np
import matplotlib.pyplot as plt
import math
import time



M0=25
M1=-25
A=12.5


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
def diffx(kx,ky,SOC):
    diffx=np.array([[2*M1*math.sin(kx), -1j*A*math.cos(kx), -1j*A*math.cos(kx)],
               [1j*A*math.cos(kx),-2*M1*math.sin(kx), 0],
                [1j*A*math.cos(kx), 0,-2*M1*math.sin(kx)]])
    return diffx
def diffy(kx,ky,SOC):
    diffy=np.array([[2*M1*math.sin(ky), -1*A*math.cos(ky), A*math.cos(ky)],
               [-1*A*math.cos(ky),-2*M1*math.sin(ky), 0],
                [A*math.cos(ky), 0,-2*M1*math.sin(ky)]])
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
SOC=np.linspace(10,15,2)
chern_change=np.zeros((len(SOC),3))
start=time.time()
for k in range(len(SOC)):
    chern_1 = 0
    chern_2 = 0
    chern_3 = 0
    for i in range(len(kx)-1):
         for j in range(len(ky)-1):
            eng0,eng1,eng2,sta0,sta1,sta2=calate(kx[i]+Delta/2,ky[j]+Delta/2,SOC[k])
            partial01x=np.dot(sta0.transpose().conj(),np.dot(diffx(kx[i]+Delta/2,ky[j]+Delta/2,SOC[k]),sta1))*np.dot(sta1.transpose().conj(),np.dot(diffy(kx[i]+Delta/2,ky[j]+Delta/2,SOC[k]),sta0))
            partial01y=np.dot(sta0.transpose().conj(),np.dot(diffy(kx[i]+Delta/2,ky[j]+Delta/2,SOC[k]),sta1))*np.dot(sta1.transpose().conj(),np.dot(diffx(kx[i]+Delta/2,ky[j]+Delta/2,SOC[k]),sta0))
            partial02x=np.dot(sta0.transpose().conj(),np.dot(diffx(kx[i]+Delta/2,ky[j]+Delta/2,SOC[k]),sta2))*np.dot(sta2.transpose().conj(),np.dot(diffy(kx[i]+Delta/2,ky[j]+Delta/2,SOC[k]),sta0))
            partial02y=np.dot(sta0.transpose().conj(),np.dot(diffy(kx[i]+Delta/2,ky[j]+Delta/2,SOC[k]),sta2))*np.dot(sta2.transpose().conj(),np.dot(diffx(kx[i]+Delta/2,ky[j]+Delta/2,SOC[k]),sta0))
            chern_1=chern_1+((partial01x-partial01y)/((eng0-eng1)**2)+(partial02x-partial02y)/((eng0-eng2)**2))*Delta**2

    for i in range(len(kx) - 1):
        for j in range(len(ky) - 1):
            eng0, eng1, eng2, sta0, sta1, sta2 = calate(kx[i]+Delta/2,ky[j]+Delta/2,SOC[k])
            partial10x = np.dot(sta1.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta0)) * np.dot(sta0.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta1))
            partial10y = np.dot(sta1.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta0)) * np.dot(sta0.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta1))
            partial12x = np.dot(sta1.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta2)) * np.dot(sta2.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta1))
            partial12y = np.dot(sta1.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta2)) * np.dot(sta2.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta1))
            chern_2 = chern_2 + ((partial10x - partial10y) / ((eng1 - eng0) ** 2) + (partial12x - partial12y)/((eng1 - eng2) ** 2))*Delta**2


    for i in range(len(kx) - 1):
        for j in range(len(ky) - 1):
            eng0, eng1, eng2, sta0, sta1, sta2 = calate(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k])
            partial20x = np.dot(sta2.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta0)) * np.dot(sta0.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta2))
            partial20y = np.dot(sta2.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta0)) * np.dot(sta0.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta2))
            partial21x = np.dot(sta2.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta1)) * np.dot(sta1.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta2))
            partial21y = np.dot(sta2.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta1)) * np.dot(sta1.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,SOC[k]), sta2))
            chern_3 = chern_3 + ((partial20x - partial20y) / ((eng2 - eng0) ** 2) + (partial21x - partial21y)/((eng2 - eng1) ** 2))*Delta**2
    chern_change[k, 0] = (1j * chern_1 / (2 * math.pi)).real
    chern_change[k, 1] = (1j * chern_2 / (2 * math.pi)).real
    chern_change[k, 2] = (1j * chern_3 / (2 * math.pi)).real
print(chern_change)
# np.save('chernnum.npy',chern_change)
# plt.scatter(SOC, chern_change[:, 0], label='chern_1')
# plt.scatter(SOC, chern_change[:, 1], label='chern_2')
# plt.scatter(SOC, chern_change[:, 2], label='chern_3')
# plt.show()
# end=time.time()
# print(end-start)
# print("最下面能带的陈数为",1j*chern_1/(2*math.pi))
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