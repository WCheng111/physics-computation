
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

"""
lattice constant a = 1 nm = 10 A
"""
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

M00 = 1
M01 = -1
M02 = 0.5
A1 = 0.5
A2 = 0.5
B1 = 0.1
C1 = 0
D1 = 0
D2 = 0
soc =1.5
Break4m = 0
Break4x = 3
socm1 = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
# socm1=np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,0],
#                   [0,0,0,0,0,1]])
socm2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
aa = 1
L = 50
V = 0
V_length = 20


def H_TISM(kx, ky, kz):
    H_TISM = (M00 + 2 * M01 * (1 - cos(kx)) + 2 * M01 * (1 - cos(ky)) + 2 * M02 * (1 - cos(kz))) * Dia_TISM + A1 * sin(
        kx) * M[2][1] + \
             A1 * sin(ky) * M[2][2] + A2 * sin(kx) * M[5][0] + A2 * sin(ky) * M[4][3] + \
             (2 * D1 * (1 - cos(kx)) - 2 * D1 * (1 - cos(ky))) * M[6][1] + D2 * sin(kx) * sin(ky) * M[6][2] + B1 * sin(
        kz) * M[2][3] + soc * socm1 + \
             Break4x * sin(kz) * M[4][1]
    return H_TISM


def H_TI(kx, ky, kz):
    H_TI = (M00 + 2 * M01 * (1 - cos(kx)) + 2 * M01 * (1 - cos(ky)) + 2 * M02 * (1 - cos(kz))) * Pauli[3][0] + A1 * sin(
        kx) * Pauli[2][1] + \
           A1 * sin(ky) * Pauli[2][2] + B1 * sin(kz) * Pauli[2][3]
    return H_TI


def H_SM(kx, ky, kz):
    H_SM = (M00 + 2 * M01 * (1 - cos(kx)) + 2 * M01 * (1 - cos(ky)) + 2 * M02 * (1 - cos(kz))) * Pauli[3][0] + A2 * sin(
        kx) * Pauli[2][0] + \
           A2 * sin(ky) * Pauli[1][3] + soc * socm2 + Break4x * sin(kz) * Pauli[1][1]
    return H_SM


ks = np.linspace(-1, 1, 201) * pi*math.sqrt(2)
ME = 2


def main():
    start = time.time()
    Eng1 = []
    Eng2 = []
    Eng3 = []
    for i in range(len(ks)):
        kx = ks[i]/sqrt(2)
        ky = ks[i]/sqrt(2)
        kz = 0
        Ham1 = H_TISM(kx, ky, kz)
        Ham2 = H_TI(kx, ky, kz)
        Ham3 = H_SM(kx, ky, kz)
        eng1, sta1 = np.linalg.eigh(Ham1)
        Eng1.append(eng1)
        eng2, sta2 = np.linalg.eigh(Ham2)
        Eng2.append(eng2)
        eng3, sta3 = np.linalg.eigh(Ham3)
        Eng3.append(eng3)

    mus = np.zeros(len(ks))
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(ks / np.pi, np.array(Eng1), 'b', lw=4)
    plt.plot(ks / np.pi, mus, 'r', linestyle='--', linewidth=2)
    plt.ylabel("E")
    plt.xlabel(r"$k_{m}$")
  #  plt.xlim(-1, 1)
    plt.grid(ls='--', c='gray')
    plt.title("TISM")


    plt.subplot(1, 3, 2)
    plt.plot(ks / np.pi, np.array(Eng2), 'b', lw=4)
    plt.plot(ks / np.pi, mus, 'r', linestyle='--', linewidth=2)
    plt.ylabel("E")
    plt.xlabel(r"$k_{m}$")
   # plt.xlim(-1, 1)
    plt.grid(ls='--', c='gray')
    plt.title("TI")

    plt.subplot(1, 3, 3)
    plt.plot(ks / np.pi, np.array(Eng3), 'b', lw=4)
    plt.plot(ks / np.pi, mus, 'r', linestyle='--', linewidth=2)
    plt.ylabel("E")
    plt.xlabel(r"$k_{m}$")
    #plt.xlim(-1, 1)
    plt.grid(ls='--', c='gray')
    plt.title("SM")
    plt.show()
    end = time.time()
    print('Running time: %s seconds' % (end - start))
    plt.savefig("%s2.jpg" % ME)
    np.save('123', Eng1)


if __name__ == '__main__':
    main()







