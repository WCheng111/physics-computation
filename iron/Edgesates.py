import kwant
from matplotlib import pyplot as plt
import tinyarray
import numpy as np
from numpy import sqrt
import time
import scipy.sparse.linalg as sla
from numpy import exp, pi, cos, sin, arccos, arcsin, sign, kron
import matplotlib.cm as cm

# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets
# from pathos.multiprocessing import Pool


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
L = 60
V = 0
V_length = 20


def make_system_TISM():
    es = 1e-3

    def system_shape(pos):
        x, = pos
        return (0 - es < x < L + es)

    def onsite(site, ky, kz):
        x, = site.pos
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


M00 = 1
M01 = -1
M02 = 0.5
A1 = 0.5
A2 = 0.5
B1 = 0.5
C1 = 0
D1 = 0
D2 = 0
soc = 0.5
Break4m = 0
Break4x = 0
edge_L = 10


def main():
    start = time.time()

    system_TISM = make_system_TISM().finalized()
    Rho_TISM = kwant.operator.Density(system_TISM)

    def cal_energy_TISM(k1, k2):
        ham_mat = system_TISM.hamiltonian_submatrix(sparse=False, params=dict(ky=k1, kz=k2))
        energies, state = np.linalg.eigh(ham_mat)
        Density_tol = np.empty(((int(len(state[:, 0]) / 6)), len(energies)), )
        den_edge_tol = np.zeros((len(energies)), )
        for i in range(len(energies)):
            Density_tol[:, i] = Rho_TISM(state[:, i])

            for j in range(len(system_TISM.sites)):
                pos_i, = system_TISM.sites[j].pos
                if 0 <= pos_i <= edge_L or L - edge_L <= pos_i <= L:
                    den_edge_tol[i] = den_edge_tol[i] + Density_tol[j, i]

        state = np.dot(state, state)
        return np.sort(energies), Density_tol

    ks = np.linspace(-1, 1, 80) * pi
    #     with Pool() as process_pool:
    #         Eng = process_pool.starmap_async(cal_energy,zip(ks,0*ks,)).get()
    Eng1 = np.empty((len(ks), 6 * (L + 1)), )
    Den1 = np.empty((len(ks), 6 * (L + 1)), )

    for i in range(len(ks)):
        k1 = ks[i]
        k2 = 0
        Eng_TISM, Den_TISM = cal_energy_TISM(k1, k2)
        Eng1[i] = Eng_TISM
        Den1[i] = Den_TISM[0]

    my_cmap = cm.get_cmap('Reds')
    norm = cm.colors.Normalize(0, 1)
    cmmapable = cm.ScalarMappable(norm, my_cmap)
    mus = np.zeros(len(ks))
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 1, 1)
    for i in range(6 * (L + 1)):
        for j in range(len(ks)):
            plt.scatter(ks[j] / np.pi, Eng1[j, i], color=my_cmap(Den1[j, i]), alpha=(Den1[j, i] / np.max(Den1)))
           # if 0.2 < Den1[j, i]:
              #  print('The energy', Eng1[j, i])
                #print('The edge density of edge state is ', Den1[j, i])
    #  plt.colorbar(cmmapable)
    plt.plot(ks / np.pi, mus, 'r', linestyle='--', linewidth=2)
    plt.ylabel("E/M0")
    plt.xlabel(r"$k_{y}$")
    plt.xlim(-1, 1)
    plt.ylim(-2,2)
    plt.title("TISM")
    plt.grid(ls='--', c='gray')
    plt.show()

    end = time.time()
    print('Running time: %s seconds' % (end - start))
    print(np.max(Den1))


if __name__ == '__main__':
    main()
