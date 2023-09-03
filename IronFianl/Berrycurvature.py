import numpy as np
import matplotlib.pyplot as plt
import math
import time
from matplotlib.colors import Normalize

M0=25
B3=-10
B=25
A=4
SOC=0.01

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

N=400
kx=np.linspace(-math.pi,math.pi,N)
ky=np.linspace(-math.pi,math.pi,N)
# Delta=2*math.pi/N
KX,KY=np.meshgrid(kx,ky)
A3=-4
Berry0=np.zeros((len(kx),len(ky)))
Berry1=np.zeros((len(kx),len(ky)))
Berry2=np.zeros((len(kx),len(ky)))

## calculate the Berry curvature of upband
for i in range(len(kx)):
     for j in range(len(ky)):
        eng0,eng1,eng2,sta0,sta1,sta2=calate(kx[i],ky[j],A3)
        partial01x=np.dot(sta2.transpose().conj(),np.dot(diffx(kx[i],ky[j],A3),sta0))*np.dot(sta0.transpose().conj(),np.dot(diffy(kx[i],ky[j],A3),sta2))
        partial01y=np.dot(sta2.transpose().conj(),np.dot(diffy(kx[i],ky[j],A3),sta0))*np.dot(sta0.transpose().conj(),np.dot(diffx(kx[i],ky[j],A3),sta2))
        partial02x=np.dot(sta2.transpose().conj(),np.dot(diffx(kx[i],ky[j],A3),sta1))*np.dot(sta1.transpose().conj(),np.dot(diffy(kx[i],ky[j],A3),sta2))
        partial02y=np.dot(sta2.transpose().conj(),np.dot(diffy(kx[i],ky[j],A3),sta1))*np.dot(sta1.transpose().conj(),np.dot(diffx(kx[i],ky[j],A3),sta2))
        Berry0[i,j]=(1j*((partial01x-partial01y)/((eng2-eng0)**2)+(partial02x-partial02y)/((eng2-eng1)**2))).real


## Calculate the Berry curvature of middleband
for i in range(len(kx)):
     for j in range(len(ky)):
        eng0,eng1,eng2,sta0,sta1,sta2=calate(kx[i],ky[j],A3)
        partial01x=np.dot(sta1.transpose().conj(),np.dot(diffx(kx[i],ky[j],A3),sta0))*np.dot(sta0.transpose().conj(),np.dot(diffy(kx[i],ky[j],A3),sta1))
        partial01y=np.dot(sta1.transpose().conj(),np.dot(diffy(kx[i],ky[j],A3),sta0))*np.dot(sta0.transpose().conj(),np.dot(diffx(kx[i],ky[j],A3),sta1))
        partial02x=np.dot(sta1.transpose().conj(),np.dot(diffx(kx[i],ky[j],A3),sta2))*np.dot(sta2.transpose().conj(),np.dot(diffy(kx[i],ky[j],A3),sta1))
        partial02y=np.dot(sta1.transpose().conj(),np.dot(diffy(kx[i],ky[j],A3),sta2))*np.dot(sta2.transpose().conj(),np.dot(diffx(kx[i],ky[j],A3),sta1))
        Berry1[i,j]=(1j*((partial01x-partial01y)/((eng1-eng0)**2)+(partial02x-partial02y)/((eng1-eng2)**2))).real
## Calculate the Berry curvature of downband
for i in range(len(kx)):
     for j in range(len(ky)):
        eng0,eng1,eng2,sta0,sta1,sta2=calate(kx[i],ky[j],A3)
        partial01x=np.dot(sta0.transpose().conj(),np.dot(diffx(kx[i],ky[j],A3),sta1))*np.dot(sta1.transpose().conj(),np.dot(diffy(kx[i],ky[j],A3),sta0))
        partial01y=np.dot(sta0.transpose().conj(),np.dot(diffy(kx[i],ky[j],A3),sta1))*np.dot(sta1.transpose().conj(),np.dot(diffx(kx[i],ky[j],A3),sta0))
        partial02x=np.dot(sta0.transpose().conj(),np.dot(diffx(kx[i],ky[j],A3),sta2))*np.dot(sta2.transpose().conj(),np.dot(diffy(kx[i],ky[j],A3),sta0))
        partial02y=np.dot(sta0.transpose().conj(),np.dot(diffy(kx[i],ky[j],A3),sta2))*np.dot(sta2.transpose().conj(),np.dot(diffx(kx[i],ky[j],A3),sta0))
        Berry2[i,j]=(1j*((partial01x-partial01y)/((eng0-eng1)**2)+(partial02x-partial02y)/((eng0-eng2)**2))).real


fig = plt.figure(figsize=(10, 12))

vmin = -10  # 最小值
vmax = 10 # 最大值

# 创建一个颜色映射
cmap = plt.get_cmap('coolwarm')
# 子图1

ax1 = fig.add_subplot(311, projection='3d')
ax1.plot_surface(KX, KY, Berry0,  cmap=cmap, vmin=vmin, vmax=vmax)
ax1.set_axis_off()
# # ax1.set_xlabel('X')
# # ax1.set_ylabel('Y')
# # ax1.set_zlabel('Z')
# # ax1.set_title('z=x^2+y^2')
#
# # 子图2
#
ax2 = fig.add_subplot(312, projection='3d')
ax2.plot_surface(KX, KY, Berry1,  cmap=cmap, vmin=vmin, vmax=vmax)
ax2.set_axis_off()
# # ax2.set_xlabel('X')
# # ax2.set_ylabel('Y')
# # ax2.set_zlabel('Z')
# # ax2.set_title('z=2(x^2+y^2)')
plt.subplots_adjust(hspace=-0.1)

# # 子图3
#
ax3 = fig.add_subplot(313, projection='3d')
ax3.plot_surface(KX, KY, Berry2,  cmap=cmap, vmin=vmin, vmax=vmax)
ax3.set_axis_off()
# # ax3.set_xlabel('X')
# # ax3.set_ylabel('Y')
# # ax3.set_zlabel('Z')
# # ax3.set_title('z=4(x^2+y^2)')
#
# 调整子图之间的垂直间距
plt.subplots_adjust(hspace=-0.1)
# plt.colorbar()

# 显示图形
plt.show()

# fig = plt.figure(figsize=(10, 12))
# # fig.set_axis_off()
# # 创建一个较大的Axes对象
# ax_big = fig.add_subplot(111, projection='3d')
#
# # 子图1
# ax1 = fig.add_axes([0.1, 0.67, 0.8, 0.25], projection='3d')
# ax1.plot_surface(KX, KY, Berry0, cmap=cmap, vmin=vmin, vmax=vmax)
# ax1.set_axis_off()
#
#
# # 子图2
# ax2 = fig.add_axes([0.1, 0.4, 0.8, 0.25], projection='3d')
# ax2.plot_surface(KX, KY, Berry1,  cmap=cmap, vmin=vmin, vmax=vmax)
# ax2.set_axis_off()
#
# # 子图3
# ax3 = fig.add_axes([0.1, 0.13, 0.8, 0.25], projection='3d')
# ax3.plot_surface(KX, KY, Berry2,  cmap=cmap, vmin=vmin, vmax=vmax)
# ax3.set_axis_off()
# plt.show()