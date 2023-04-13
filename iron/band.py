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

M00 = -1
M01 = 1
M02 = 1
A1 = 0.5
A2 =0.5
A3=0
B1 =0
C1 = 0
D1 = 0
D2 = 0
# soc = 2
Break4m = 0
Break4x = 0
socm1 = np.array([[-1, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0], [0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

def cal1(ks1, soc):
    Eng=[]
    for i in range(len(ks1)):
        kx=0
        ky=ks1[i]
        kz=0
        H_TISM = (M00 + 2 * M01 * (1 - cos(kx)) + 2 * M01 * (1 - cos(ky)) + 2 * M02 * (1 - cos(kz))) * Dia_TISM + A1 * sin(
            kx) * M[2][1] + \
                 A1 * sin(ky) * M[2][2] + A2 * sin(kx) * M[5][0] + A3 * sin(ky) * M[4][3] + \
                 (2 * D1 * (1 - cos(kx)) - 2 * D1 * (1 - cos(ky))) * M[6][1] + D2 * sin(kx) * sin(ky) * M[6][2] + B1 * sin(
            kz) * M[2][3] + soc * socm1 \
                 +Break4x * sin(kz) * M[4][1]
        eng, sta=np.linalg.eigh(H_TISM)
        Eng.append(eng)
        Eng1=np.array(Eng)
        Eng2=[]
    for j in range(len(ks1)):
        Eng2.append(Eng[j][0])
    return np.array(Eng2)

def cal2(ks1, soc):
    Eng=[]
    for i in range(len(ks1)):
        kx=0
        ky=ks1[i]
        kz=0
        H_TISM = (M00 + 2 * M01 * (1 - cos(kx)) + 2 * M01 * (1 - cos(ky)) + 2 * M02 * (1 - cos(kz))) * Dia_TISM + A1 * sin(
            kx) * M[2][1] + \
                 A1 * sin(ky) * M[2][2] + A2 * sin(kx) * M[5][0] + A3 * sin(ky) * M[4][3] + \
                 (2 * D1 * (1 - cos(kx)) - 2 * D1 * (1 - cos(ky))) * M[6][1] + D2 * sin(kx) * sin(ky) * M[6][2] + B1 * sin(
            kz) * M[2][3] + soc * socm1 \
                 +Break4x * sin(kz) * M[4][1]
        eng, sta=np.linalg.eigh(H_TISM)
        Eng.append(eng)
        Eng1=np.array(Eng)
        Eng2=[]
    for j in range(len(ks1)):
        Eng2.append(Eng[j][3])
    return np.array(Eng2)

def cal3(ks1, soc):
    Eng=[]
    for i in range(len(ks1)):
        kx=0
        ky=ks1[i]
        kz=0
        H_TISM = (M00 + 2 * M01 * (1 - cos(kx)) + 2 * M01 * (1 - cos(ky)) + 2 * M02 * (1 - cos(kz))) * Dia_TISM + A1 * sin(
            kx) * M[2][1] + \
                 A1 * sin(ky) * M[2][2] + A2 * sin(kx) * M[5][0] + A3 * sin(ky) * M[4][3] + \
                 (2 * D1 * (1 - cos(kx)) - 2 * D1 * (1 - cos(ky))) * M[6][1] + D2 * sin(kx) * sin(ky) * M[6][2] + B1 * sin(
            kz) * M[2][3] + soc * socm1 \
                 +Break4x * sin(kz) * M[4][1]
        eng, sta=np.linalg.eigh(H_TISM)
        Eng.append(eng)
        Eng1=np.array(Eng)
        Eng2=[]
    for j in range(len(ks1)):
        Eng2.append(Eng[j][5])
    return np.array(Eng2)







kx = np.linspace(-1, 1, 201) * pi
# print(cal(ks, 2)[])
int_soc = 0
m=[]
for i in range(len(kx)):
    m.append(0)

fig, ax = plt.subplots()
plt.xlim(-2.1,2.1)
plt.ylim(-1.5,1.5)
plt.xlabel('kx')
plt.ylabel('E')
line1, =ax.plot(kx, cal1(kx, int_soc), color='dodgerblue', linewidth=4.0)
line2, =ax.plot(kx, cal2(kx, int_soc), color='red',linewidth=4.0)
line3, =ax.plot(kx, cal3(kx, int_soc),color='dodgerblue',linewidth=4.0)
line4, =ax.plot(kx, m, color='black',linewidth=3.0,linestyle='--')
ax.set_xlabel('kx')


# fig.subplots_adjust(left=0.25, bottom=0.25)
#
#
# axsoc = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# soc_slider = Slider(
#     ax=axsoc,
#     label='soc',
#     valmin=-5,
#     valmax=5,
#     valinit=int_soc,
# )
#
#
# def update(val):
#      line1.set_ydata(cal1(ks, soc_slider.val))
#      line2.set_ydata(cal2(ks, soc_slider.val))
#      line3.set_ydata(cal3(ks, soc_slider.val))
#      fig.canvas.draw_idle()
#
#
# soc_slider.on_changed(update)
#
#
# resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Reset', hovercolor='0.975')
#
#
# def reset(event):
#     soc_slider.reset()
# button.on_clicked(reset)

plt.show()