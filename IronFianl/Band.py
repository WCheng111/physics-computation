import numpy as np
import matplotlib.pyplot as plt
import math
import time



M0=25
B3=-10
B=25
A=4
SOC=30

def H(kx,ky,SOC):
    H=np.array([[M0+2*B3*(1-math.cos(kx))+2*B*(1-math.cos(ky)), -A*(math.sin(kx)-1j*math.sin(ky)), -A*(math.sin(kx)-1j*math.sin(ky))],
               [-A*(math.sin(kx)+1j*math.sin(ky)),-(M0+2*B3*(1-math.cos(kx))+2*B*(1-math.cos(ky))), 0],
                [-A*(math.sin(kx)-1j*math.sin(ky)), 0,-(M0+2*B3*(1-math.cos(kx))+2*B*(1-math.cos(ky)))+SOC]])
    return H
## 需要比对一下TI的gap是多少

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
kx=np.linspace(0,1,201)*math.pi
kxx=np.linspace(-1,1,201)*math.pi
E=np.zeros((2*len(kx),3))
Ex=np.zeros((len(kxx),3))
for i in range(len(kxx)):
    Ex[i,:]=calate(kxx[i],0,SOC)[0:3]
for i in range(len(kx)):
    E[i,:]=calate(math.pi-kx[i],math.pi-kx[i],SOC)[0:3]
for i in range(len(kx)):
    E[i+len(kx),:]=calate(kx[i],0,SOC)[0:3]

x=np.linspace(-1,1,2*len(kx))*math.pi
plt.plot(x,E[:,0],label='band1',color='black')
plt.plot(x,E[:,1],label='band2',color='black')
plt.plot(x,E[:,2],label='band3',color='black')
# plt.axvline(x=0, ymin=0, ymax=1.2, color='r', linestyle='-',linewidth=1)

# plt.xlabel('kx')



# plt.title(f'soc={SOC}')
plt.ylabel('Energy')


# plt.plot(kxx,Ex[:,2],label='up band',color='red',linewidth=3)
# plt.plot(kxx,Ex[:,1],label='middle band',color='green',linewidth=3)
# plt.plot(kxx,Ex[:,0],label='bottom band',color='blue',linewidth=3)
# plt.axvline(x=-math.pi, ymin=0, ymax=1.2, color='black', linestyle='--',linewidth=2)
# plt.axvline(x=0, ymin=0, ymax=1.2, color='black', linestyle='--',linewidth=2)
# plt.axvline(x=math.pi, ymin=0, ymax=1.2, color='black', linestyle='--',linewidth=2)
# plt.legend(loc='upper center', bbox_to_anchor=(0.2, 1))
# plt.xticks([])
# plt.text(-math.pi, -32, r'$Z$', ha='center', fontsize=15)
# plt.text(0, -32, r'$\Gamma$', ha='center', fontsize=15)
# plt.text(math.pi, -32, r'$Z$', ha='center', fontsize=15)
# plt.ylabel('Energy')


plt.show()