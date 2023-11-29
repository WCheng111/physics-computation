import numpy as np
import matplotlib.pyplot as plt
import math
import time



M01=50
B31=-25
B1=80
M02=-10
B32=5
B2=-80
A1=5
A2=50
SOC=10

def H(kz,ky,SOC):
    H=np.array([[M01+2*B31*(1-math.cos(kz))+2*B1*(1-math.cos(ky)), A1*math.sin(kz)+A2*1j*math.sin(ky), A1*math.sin(kz)+A2*1j*math.sin(ky)],
               [A1*math.sin(kz)-A2*1j*math.sin(ky),(M02+2*B32*(1-math.cos(kz))+2*B2*(1-math.cos(ky)))-SOC/2, 0],
                [A1*math.sin(kz)-A2*1j*math.sin(ky), 0,(M02+2*B32*(1-math.cos(kz))+2*B2*(1-math.cos(ky)))+SOC/2]])
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
print(calate(0,0,30))
##Plot the band along the Y'-Z-gamma
# Zgamma=np.linspace(0,1,201)*math.pi
# Yz=np.linspace(-1,0,201)*math.pi
# kx=np.linspace(-1,1,2*len(Zgamma))
# E=np.zeros((2*len(Zgamma),3))
# for i in range(len(Yz)):
#     E[i,:]=calate(math.pi,Yz[i],SOC)[0:3]
# for j in range(len(Zgamma)):
#     E[len(Yz)+j,:] = calate(math.pi-Zgamma[j], 0, SOC)[0:3]
#
# plt.plot(kx,E[:,0],label='band1',color='black')
# plt.plot(kx,E[:,1],label='band2',color='black')
# plt.plot(kx,E[:,2],label='band3',color='black')
# plt.axvline(0, color='k', linestyle='--',)
# plt.axvline(-1, color='k', linestyle='--',)
# plt.axvline(1, color='k', linestyle='--',)
# plt.text(0,-120, r'$Z$',size=20)
# plt.text(1,-120, r'$\Gamma$',size=20)
# plt.text(-1,-120, r'$Y‘$',size=20)
# plt.ylim(-100,100)
# plt.xticks([])
# # plt.xlabel('kz')
# plt.title(f'soc={SOC}')
# plt.ylabel('Energy')
# plt.show()


##Plot the band along Y-Gamma-Z
# Ygamma=np.linspace(0,1,201)*math.pi
# gammaz=np.linspace(0,1,201)*math.pi
# E2=np.zeros((2*len(Ygamma),3))
# kx=np.linspace(-1,1,2*len(Ygamma))
# for i in range(len(Ygamma)):
#     E2[i, :] = calate(0, -math.pi+Ygamma[i], SOC)[0:3]
# for j in range(len(gammaz)):
#     E2[len(Ygamma) + j, :] = calate(gammaz[j], 0, SOC)[0:3]
# plt.plot(kx,E2[:,0],label='band1',color='black')
# plt.plot(kx,E2[:,1],label='band2',color='black')
# plt.plot(kx,E2[:,2],label='band3',color='black')
# plt.axvline(0, color='k', linestyle='--',)
# plt.axvline(-1, color='k', linestyle='--',)
# plt.axvline(1, color='k', linestyle='--',)
# plt.text(0,-120, r'$\Gamma$',size=20)
# plt.text(1,-120, r'$Z$',size=20)
# plt.text(-1,-120, r'$Y$',size=20)
# plt.xticks([])
# plt.ylim(-100,100)
# # plt.xlabel('kz')
# plt.title(f'soc={SOC}')
# plt.ylabel('Energy')
# plt.show()

##plot the band along the gamma-Z-gamma

Zgamma=np.linspace(0,1,201)*math.pi
Yz=np.linspace(-1,0,201)*math.pi
kx=np.linspace(-1,1,2*len(Zgamma))
E=np.zeros((2*len(Zgamma),3))
for i in range(len(Yz)):
    E[i,:]=calate(Zgamma[i],0,SOC)[0:3]
for j in range(len(Zgamma)):
    E[len(Yz)+j,:] = calate(math.pi-Zgamma[j], 0, SOC)[0:3]

plt.plot(kx,E[:,0],label='band1',color='black')
plt.plot(kx,E[:,1],label='band2',color='black')
plt.plot(kx,E[:,2],label='band3',color='black')
plt.axvline(0, color='k', linestyle='--',)
plt.axvline(-1, color='k', linestyle='--',)
plt.axvline(1, color='k', linestyle='--',)
plt.text(0,-120, r'$Z$',size=20)
plt.text(1,-120, r'$\Gamma$',size=20)
plt.text(-1,-120, r'$\Gamma$',size=20)
plt.ylim(-100,100)
plt.xticks([])
# plt.xlabel('kz')
plt.title(f'soc={SOC}')
plt.ylabel('Energy')
plt.show()



#Plot the band along the gY-Gamma,Z