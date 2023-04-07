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
from numpy import invert as inv
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

PM=list(range(2))
PM[0]=np.array([[1,0],[0,0]])
PM[1]=np.array([[0,0],[0,1]])

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
socm1 = np.array([[-1, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0], [0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
deltamatr_1=np.kron(P[1],np.kron(G[0],1j*P[2]))
deltamatr_2=np.kron(P[2],np.kron(G[0],1j*P[2]))


kx=pi/2
ky=pi/2
kz=0
soc=0
M00 = -1
M01 = 1
M02 = 1
A1 = 0.5
A2 =0.5
B1 =0
theta=pi/2
Delta=0.5


break4 = 0.5
H_TISMe = (M00 + 2 * M01 * (1 - cos(kx)) + 2 * M01 * (1 - cos(ky)) + 2 * M02 * (1 - cos(kz))) * Dia_TISM + A1 * sin(
            kx) * M[2][1] + A1 * sin(ky) * M[2][2] + A2 * sin(kx) * M[5][0] + A2 * sin(ky) * M[4][3] + \
                  + B1 * sin(kz) * M[2][3] + soc * socm1 +break4* sin(kz) * M[4][1]
H_TISMh=(M00 + 2 * M01 * (1 - cos(kx)) + 2 * M01 * (1 - cos(ky)) + 2 * M02 * (1 - cos(kz))) * Dia_TISM + A1 * sin(
            kx) * M[2][1] -A1 * sin(ky) * M[2][2] + A2 * sin(kx) * M[5][0] - A2 * sin(ky) * M[4][3] + \
                  + B1 * sin(kz) * M[2][3] + soc * socm1 +break4* sin(kz) * M[4][1]
H_delta=np.kron(PM[0],H_TISMe)-np.kron(PM[1],H_TISMh)+Delta*cos(theta)*deltamatr_1+Delta*sin(theta)*deltamatr_2




print(H_delta)
Trans=[[[0 for i in range(4)] for j in range(9)]for k in range(4)]
for k in range(4):
    for j in range(9):
        for i in range(4):
            Trans[k][j][i]=np.kron(P[k],np.kron(G[j],P[i]))



for k in range(4):
    for j in range(9):
        for i in range(4):
            if np.linalg.det(Trans[k][j][i])!=0:
                # if (np.dot(np.dot(Trans[k][j][i],H_delta),np.linalg.inv(Trans[k][j][i]))==-H_delta).all():
                if(Trans[k][j][i]@H_delta@np.linalg.inv(Trans[k][j][i])==-H_delta).all():
                                print(k,j,i)

