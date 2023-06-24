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
a = 1
L = 80
V = 0
V_length = 20
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
kz=0
def make_system_TISM():  # a = 1为晶格常数, L, W 为长宽


    lat = kwant.lattice.square(a,norbs=6)  # 创建2D正方晶格

    syst = kwant.Builder(kwant.TranslationalSymmetry([0, a]))  # 建立中心体系，利用平移对称性使y方向为无穷大，可视为周期边界

    syst[(lat(x, 0) for x in range(L))] = (M00  + 2 * M02 * (1 - cos(kz))) * Dia_TISM + B1 * sin(kz) * M[2][3] \
                 + 4 * M01 * Dia_TISM + \
                +Break4x * sin(kz) * M[4][1] + soc * socm1
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] =  -M01 * Dia_TISM + A1 / (2j) * M[2][1] + A2 / (2j) * M[5][0]
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = -M01 * Dia_TISM + A1 / (2j) * M[2][2] + A2 / (2j) * M[4][3]

    kwant.plot(syst, dpi=300)  # 系统示意图，通过图像可以看出有没有写错
    syst = syst.finalized()  # 结束体系的制作。这个语句不可以省。这个语句是把Builder对象转换成可计算的对象。
    return syst
kwant.plotter.bands(make_system_TISM(), momenta=np.linspace(-pi, pi, 201), show=False, dpi=600)
plt.xlabel("momentum [(lattice constant)$^{-1}$]")
plt.ylabel("Energy [$t$]")
plt.show()
# def main():
#     start = time.time()
#
#     system_TISM = make_system_TISM().finalized()
#     # system_TI = make_system_TI().finalized()
#     # system_SM = make_system_SM().finalized()
#
#     def cal_energy_TISM(k1, k2):
#         ham_mat = system_TISM.hamiltonian_submatrix(sparse=False, params=dict(ky=k1, kz=k2))
#         energies, state = np.linalg.eigh(ham_mat)
#         return np.sort(energies)
#
#     # def cal_energy_TI(k1, k2):
#     #     ham_mat = system_TI.hamiltonian_submatrix(sparse=False, params=dict(ky=k1, kz=k2))
#     #     energies, state = np.linalg.eigh(ham_mat)
#     #     return np.sort(energies)
#     #
#     # def cal_energy_SM(k1, k2):
#     #     ham_mat = system_SM.hamiltonian_submatrix(sparse=False, params=dict(ky=k1, kz=k2))
#     #     energies, state = np.linalg.eigh(ham_mat)
#     #     return np.sort(energies)
#     #
#     # ks = np.linspace(-pi, pi, 101)
#     # #     with Pool() as process_pool:
#     # #         Eng = process_pool.starmap_async(cal_energy,zip(ks,0*ks,)).get()
#     # Eng1 = []
#     # Eng2 = []
#     # Eng3 = []
#     # for i in range(len(ks)):
#     #     k1 = ks[i]
#     #     k2 =0
#     #     Eng_TISM = cal_energy_TISM(k1, k2)
#     #     # Eng_TI = cal_energy_TI(k1, k2)
#     #     # Eng_SM = cal_energy_SM(k1, k2)
#     #     Eng1.append(Eng_TISM)
#     #     # Eng2.append(Eng_TI)
#     #     # Eng3.append(Eng_SM)
#     #
#     # mus = np.zeros(len(ks))
#     # plt.figure(figsize=(16, 8))
#     # plt.subplot(1, 3, 1)
#     # plt.plot(ks , np.array(Eng1), 'b', lw=4)
#     # plt.plot(ks , mus, 'r', linestyle='--', linewidth=2)
#     # plt.ylabel("E/M0")
#     # plt.xlabel(r"$k_{y}$")
#     # plt.xlim(-pi, pi)
#     # # plt.ylim(-2,2)
#     # plt.title("TISM")
#     # plt.grid(ls='--', c='gray')
#     # #plt.show
#     #
#     # # plt.subplot(1, 3, 2)
#     # # plt.plot(ks , np.array(Eng2), 'b', lw=4)
#     # # plt.plot(ks , mus, 'r', linestyle='--', linewidth=2)
#     # # plt.ylabel("E/M0")
#     # # plt.xlabel(r"$k_{y}$")
#     # # plt.xlim(-pi, pi)
#     # # plt.title("TI")
#     # # # plt.ylim(-2,2)
#     # # plt.grid(ls='--', c='gray')
#     # # #plt.show
#     # #
#     # # plt.subplot(1, 3, 3)
#     # # plt.plot(ks , np.array(Eng3), 'b', lw=4)
#     # # plt.plot(ks , mus, 'r', linestyle='--', linewidth=2)
#     # # plt.ylabel("E/M0")
#     # # plt.xlabel(r"$k_{y}$")
#     # # plt.xlim(-pi,pi)
#     # # plt.title("SM")
#     # # # plt.ylim(-2,2)
#     # # plt.grid(ls='--', c='gray')
#     # plt.show()
#     kwant.plotter.bands( make_system_TISM(), momenta=np.linspace(-pi, pi, 201), show=False, dpi=600)
#     end = time.time()
#     print('Running time: %s seconds' % (end - start))
#
#
# if __name__ == '__main__':
#     main()

