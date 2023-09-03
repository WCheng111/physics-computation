import numpy as np
import matplotlib.pyplot as plt
import math
import time
import multiprocessing


M0=25
B3=-10
B=25
A=4
SOC=30

def H(kx,ky,A3):
    H=np.array([[M0+2*B3*(1-math.cos(kx))+2*B*(1-math.cos(ky)), -A*(math.sin(kx)-1j*math.sin(ky)), -A3*math.sin(kx)+1j*A*math.sin(ky)],
               [-A*(math.sin(kx)+1j*math.sin(ky)),-(M0+2*B3*(1-math.cos(kx))+2*B*(1-math.cos(ky))), 0],
                [-A3*math.sin(kx)-1j*A*math.sin(ky), 0,-(M0+2*B3*(1-math.cos(kx))+2*B*(1-math.cos(ky)))+SOC]])
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
# print(diffx(0,0))
#print(np.dot(sta0,np.dot(diffx(0,0),sta0)))
# print(np.conjugate(sta0))
N=400
kx=np.linspace(-math.pi,math.pi,N)
ky=np.linspace(-math.pi,math.pi,N)
Delta=2*math.pi/N


def cal_chern(A3):
    chern_change = np.zeros((1, 3))
    chern_1 = 0
    chern_2 = 0
    chern_3 = 0
    for i in range(len(kx)-1):
         for j in range(len(ky)-1):
            eng0,eng1,eng2,sta0,sta1,sta2=calate(kx[i]+Delta/2,ky[j]+Delta/2,A3)
            partial01x=np.dot(sta0.transpose().conj(),np.dot(diffx(kx[i]+Delta/2,ky[j]+Delta/2,A3),sta1))*np.dot(sta1.transpose().conj(),np.dot(diffy(kx[i]+Delta/2,ky[j]+Delta/2,A3),sta0))
            partial01y=np.dot(sta0.transpose().conj(),np.dot(diffy(kx[i]+Delta/2,ky[j]+Delta/2,A3),sta1))*np.dot(sta1.transpose().conj(),np.dot(diffx(kx[i]+Delta/2,ky[j]+Delta/2,A3),sta0))
            partial02x=np.dot(sta0.transpose().conj(),np.dot(diffx(kx[i]+Delta/2,ky[j]+Delta/2,A3),sta2))*np.dot(sta2.transpose().conj(),np.dot(diffy(kx[i]+Delta/2,ky[j]+Delta/2,A3),sta0))
            partial02y=np.dot(sta0.transpose().conj(),np.dot(diffy(kx[i]+Delta/2,ky[j]+Delta/2,A3),sta2))*np.dot(sta2.transpose().conj(),np.dot(diffx(kx[i]+Delta/2,ky[j]+Delta/2,A3),sta0))
            chern_1=chern_1+((partial01x-partial01y)/((eng0-eng1)**2)+(partial02x-partial02y)/((eng0-eng2)**2))*Delta**2

    for i in range(len(kx) - 1):
        for j in range(len(ky) - 1):
            eng0, eng1, eng2, sta0, sta1, sta2 = calate(kx[i]+Delta/2,ky[j]+Delta/2,A3)
            partial10x = np.dot(sta1.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta0)) * np.dot(sta0.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta1))
            partial10y = np.dot(sta1.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta0)) * np.dot(sta0.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta1))
            partial12x = np.dot(sta1.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta2)) * np.dot(sta2.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta1))
            partial12y = np.dot(sta1.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta2)) * np.dot(sta2.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta1))
            chern_2 = chern_2 + ((partial10x - partial10y) / ((eng1 - eng0) ** 2) + (partial12x - partial12y)/((eng1 - eng2) ** 2))*Delta**2

    for i in range(len(kx) - 1):
        for j in range(len(ky) - 1):
            eng0, eng1, eng2, sta0, sta1, sta2 = calate(kx[i]+Delta/2, ky[j]+Delta/2,A3)
            partial20x = np.dot(sta2.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta0)) * np.dot(sta0.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta2))
            partial20y = np.dot(sta2.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta0)) * np.dot(sta0.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta2))
            partial21x = np.dot(sta2.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta1)) * np.dot(sta1.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta2))
            partial21y = np.dot(sta2.transpose().conj(), np.dot(diffy(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta1)) * np.dot(sta1.transpose().conj(), np.dot(diffx(kx[i]+Delta/2, ky[j]+Delta/2,A3), sta2))
            chern_3 = chern_3 + ((partial20x - partial20y) / ((eng2 - eng0) ** 2) + (partial21x - partial21y)/((eng2 - eng1) ** 2))*Delta**2
    chern_change[0,0] = (1j * chern_1 / (2 * math.pi)).real
    chern_change[0,1] = (1j * chern_2 / (2 * math.pi)).real
    chern_change[0,2] = (1j * chern_3 / (2 * math.pi)).real
    return chern_change


def main():
    start=time.time()
    A3=np.linspace(-4,4,150)
    num_processes = multiprocessing.cpu_count()  # 使用可用的CPU核心数
    print(num_processes)
    with multiprocessing.Pool(num_processes) as pool:
        chern = pool.map(cal_chern, A3)
    np.save('chernnumber.npy',chern)
    end=time.time()
    print(end-start)

if __name__ == '__main__':
    main()
end=time.time()