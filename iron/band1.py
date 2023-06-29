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

P = list(range(4))
P[0] = np.array([[1, 0], [0, 1]])
P[1] = np.array([[0, 1], [1, 0]])
P[2] = np.array([[0, -1j], [1j, 0]])
P[3] = np.array([[1, 0], [0, -1]])

M = [[0 for i in range(4)] for j in range(9)]
Pauli = [[0 for i in range(4)] for j in range(4)]

for i in range(9):
    for j in range(4):
        M[i][j] = np.kron(G[i], P[j])
for i in range(4):
    for j in range(4):
        Pauli[i][j] = np.kron(P[i], P[j])

Dia_TISM = np.array(
    [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0], [0, 0, 0, -1, 0, 0], [0, 0, 0, 0, -1, 0],
     [0, 0, 0, 0, 0, -1]])
PZ=np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]])
dZ=np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1]])

M00 = 50
M01 = 80
M02 = -25
M10=-10
M11=-80
M12=5
A1 = 50
A2 =50
A3=0
B1 =5
C1 = 0
D1 = 0.5
D2 = 0.5
# soc = 2
Break4m = 5
Break4x = 5
socm1 = np.array([[-1, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0], [0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
soc=10



def H_iron(kx, ky, kz):
    kx=kx
    ky=ky
    kz=kz
    H_iron= (M00 + 2 * M01 * (1 - cos(kx)) + 2 * M01 * (1 - cos(ky)) + 2 * M02 * (1 - cos(kz))) * PZ+\
            (M10 + 2 * M11 * (1 - cos(kx)) + 2 * M11 * (1 - cos(ky)) + 2 * M12 * (1 - cos(kz)))*dZ + A1 * sin(
        kx) * M[2][1] + \
             A1 * sin(ky) * M[2][2] + A2 * sin(kx) * M[5][0] + A2 * sin(ky) * M[4][3] + \
             (2 * D1 * (1 - cos(kx)) - 2 * D1 * (1 - cos(ky))) * M[6][1] + D2 * sin(kx) * sin(ky) * M[6][2] + B1 * sin(
        kz) * M[2][3] + soc * socm1 \
             + Break4x * sin(kz) * M[4][1]
    return H_iron

kz = np.linspace(-1, 1, 201) *pi
E= []
for i in range(len(kz)):
        Energy, wave=np.linalg.eigh(H_iron(kz[i],0, 0))
        E.append(Energy)
plt.plot(kz, E, color="black")
print(M[2][2])
plt.xlabel('kz')
plt.ylabel('Energy')
plt.show()


