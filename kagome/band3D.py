import matplotlib
matplotlib.use('Qt5Agg')  # 或者使用 'Qt5Agg'

from matplotlib import pyplot as plt
import numpy as np
from numpy import sqrt
from mpl_toolkits.mplot3d import Axes3D
G = list(range(9))
G[0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
G[1] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
G[2] = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
G[3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
G[4] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
G[5] = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
G[6] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
G[7] = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
G[8] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])

mu=0
t=1

def H_kagome(kx, ky):
    k1 = kx
    k2 = kx / 2 + (sqrt(3) / 2) * ky
    k3 = -kx / 2 + (sqrt(3) / 2) * ky
    H_kagome =mu*G[0]-2*t*np.cos(k1/2)*G[1]-2*t*np.cos(k2/2)*G[4]-2*t*np.cos(k3/2)*G[6]
    return H_kagome

kx = np.linspace(-2, 2, 201)
ky=np.linspace(-2, 2, 201)
KX, KY = np.meshgrid(kx, ky)
E= np.zeros((201,201))
for i in range(len(KX)):
    for j in range(len(KY)):
        Energy,wave=np.linalg.eigh(H_kagome(KX[i][j],KY[i][j]))
        E[i,j,:] =np.sort(np.real(Energy[:]))

fig = plt.figure()
ax1 = Axes3D(fig)
for b in range(3):
    ax1.plot_surface(KX,KY,E[:,:,b],rstride = 1, cstride = 1,cmap='rainbow')





