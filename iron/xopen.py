import kwant
from matplotlib import pyplot as plt
import tinyarray
import numpy as np
from numpy import sqrt
import time
import scipy.sparse.linalg as sla
from numpy import exp, pi, cos, sin, arccos, arcsin, sign, kron
import matplotlib.cm as cm
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from pathos.multiprocessing import Pool

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

socm1 = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])
# socm1=np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,1,0],
#                   [0,0,0,0,0,1]])
socm2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
aa = 1
L = 50
V = 0
V_length = 20


def make_system_TISM():
    es = 1e-3

    def system_shape(pos):
        x, = pos
        return (0 - es < x < L + es)

    def onsite(site, ky, kz):
        x, = site.pos
        if x < V_length + es:
            Vp = -V
        else:
            Vp = 0
        return (M00 + 2 * M01 * (1 - cos(ky)) + 2 * M02 * (1 - cos(kz))) * Dia_TISM + B1 * sin(kz) * M[2][3] \
               + A1 * sin(ky) * M[2][2] + A2 * sin(ky) * M[4][3] + 2 * M01 * Dia_TISM + \
               +Break4x * sin(kz) * M[4][1] + soc * socm1

    def hopping_x(site1, site2):
        return -M01 * Dia_TISM + A1 / (2j) * M[2][1] + A2 / (2j) * M[5][0]

    system = kwant.Builder()
    lat = kwant.lattice.chain(aa, norbs=6)
    system[lat.shape(system_shape, (0,))] = onsite
    system[kwant.builder.HoppingKind((1,), lat, lat)] = hopping_x

    return system


def make_system_TI():
    es = 1e-3

    def system_shape(pos):
        x, = pos
        return (0 - es < x < L + es)

    def onsite(site, ky, kz):
        x, = site.pos

        return (M00 + 2 * M02 * (1 - cos(kz)) + 2 * M01 * (1 - cos(ky))) * Pauli[3][0] + \
               A1 * sin(ky) * Pauli[2][2] + 2 * M01 * Pauli[3][0] + B1 * sin(kz) * Pauli[2][3]

    def hopping_x(site1, site2):
        return -M01 * Pauli[3][0] + A1 / (2j) * Pauli[2][1]

    system = kwant.Builder()
    lat = kwant.lattice.chain(aa, norbs=4)
    system[lat.shape(system_shape, (0,))] = onsite
    system[kwant.builder.HoppingKind((1,), lat, lat)] = hopping_x

    return system


def make_system_SM():
    es = 1e-3

    def system_shape(pos):
        x, = pos
        return (0 - es < x < L + es)

    def onsite(site, ky, kz):
        x, = site.pos

        return (M00 + 2 * M02 * (1 - cos(kz)) + 2 * M01 * (1 - cos(ky))) * Pauli[3][0] + \
               A2 * sin(ky) * Pauli[1][3] + 2 * M01 * Pauli[3][0] + Break4x * Pauli[1][1] * sin(kz) + soc * socm2
        # (M+2*B1+2*B2*(1-cos(kx))+2*B2*(1-cos(ky)))*sigma_0z+A2*sin(kx)*sigma_xx+A2*sin(ky)*sigma_yx+Vp*sigma_00

    def hopping_x(site1, site2):
        return -M01 * Pauli[3][0] + A2 / (2j) * Pauli[2][0]

    system = kwant.Builder()
    lat = kwant.lattice.chain(aa, norbs=4)
    system[lat.shape(system_shape, (0,))] = onsite
    system[kwant.builder.HoppingKind((1,), lat, lat)] = hopping_x

    return system


M00 = 1
M01 = -1
M02 = 0.5
A1 = 0.5
A2 = 0.5
B1 = 5
C1 = 0
D1 = 0
D2 = 0
soc =0.5

Break4m = 0
Break4x = 0


def main():
    start = time.time()

    system_TISM = make_system_TISM().finalized()
    system_TI = make_system_TI().finalized()
    system_SM = make_system_SM().finalized()

    def cal_energy_TISM(k1, k2):
        ham_mat = system_TISM.hamiltonian_submatrix(sparse=False, params=dict(ky=k1, kz=k2))
        energies, state = np.linalg.eigh(ham_mat)
        return np.sort(energies)

    def cal_energy_TI(k1, k2):
        ham_mat = system_TI.hamiltonian_submatrix(sparse=False, params=dict(ky=k1, kz=k2))
        energies, state = np.linalg.eigh(ham_mat)
        return np.sort(energies)

    def cal_energy_SM(k1, k2):
        ham_mat = system_SM.hamiltonian_submatrix(sparse=False, params=dict(ky=k1, kz=k2))
        energies, state = np.linalg.eigh(ham_mat)
        return np.sort(energies)

    ks = np.linspace(-pi, pi, 101)
    #     with Pool() as process_pool:
    #         Eng = process_pool.starmap_async(cal_energy,zip(ks,0*ks,)).get()
    Eng1 = []
    Eng2 = []
    Eng3 = []
    for i in range(len(ks)):
        k1 = ks[i]
        k2 =0
        Eng_TISM = cal_energy_TISM(k1, k2)
        Eng_TI = cal_energy_TI(k1, k2)
        Eng_SM = cal_energy_SM(k1, k2)
        Eng1.append(Eng_TISM)
        Eng2.append(Eng_TI)
        Eng3.append(Eng_SM)

    mus = np.zeros(len(ks))
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 3, 1)
    plt.plot(ks , np.array(Eng1), 'b', lw=4)
    plt.plot(ks , mus, 'r', linestyle='--', linewidth=2)
    plt.ylabel("E/M0")
    plt.xlabel(r"$k_{y}$")
    plt.xlim(-pi, pi)
    # plt.ylim(-2,2)
    plt.title("TISM")
    plt.grid(ls='--', c='gray')
    #plt.show

    plt.subplot(1, 3, 2)
    plt.plot(ks , np.array(Eng2), 'b', lw=4)
    plt.plot(ks , mus, 'r', linestyle='--', linewidth=2)
    plt.ylabel("E/M0")
    plt.xlabel(r"$k_{y}$")
    plt.xlim(-pi, pi)
    plt.title("TI")
    # plt.ylim(-2,2)
    plt.grid(ls='--', c='gray')
    #plt.show

    plt.subplot(1, 3, 3)
    plt.plot(ks , np.array(Eng3), 'b', lw=4)
    plt.plot(ks , mus, 'r', linestyle='--', linewidth=2)
    plt.ylabel("E/M0")
    plt.xlabel(r"$k_{y}$")
    plt.xlim(-pi,pi)
    plt.title("SM")
    # plt.ylim(-2,2)
    plt.grid(ls='--', c='gray')
    plt.show()

    end = time.time()
    print('Running time: %s seconds' % (end - start))


if __name__ == '__main__':
    main()
