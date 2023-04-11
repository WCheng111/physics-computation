import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import kwant
from matplotlib import pyplot as plt
import tinyarray
import numpy as np
from numpy import sqrt
import time
import scipy.sparse.linalg as sla
from numpy import exp, pi, cos, sin, arccos, arcsin, sign, kron
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from pathos.multiprocessing import Pool
import math
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
# The parametrized function to be plotted
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
    H_kagome =mu*G[0]-2*t*cos(k1/2)*G[1]-2*t*cos(k2/2)*G[4]-2*t*cos(k3/2)*G[6]
    return H_kagome

kx = np.linspace(-2, 2, 201) * pi
ky=np.linspace(-2, 2, 201) * pi
KX, KY = np.meshgrid(kx, ky)
E= np.zeros((201,201,3))
E2= np.zeros((5,5,2))
for i in range(len(KX)):
    for j in range(len(KY)):
        Energy,wave=np.linalg.eigh(H_kagome(KX[i][j],KY[i][j]))
        E[i,j,:] =np.sort(np.real(Energy[:]))

print(E2)
fig = plt.figure()
ax = Axes3D(fig)
for b in range(3):
    #ax1.plot_surface(K_X,K_Y,E[:,:,b],rstride = 1, cstride = 1,cmap='rainbow')
    ax.plot_surface(KX,KY,E[:,:,b],cmap='viridis')

plt.show()


