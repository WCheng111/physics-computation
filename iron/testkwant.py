
import kwant
import numpy as np
import math
from math import pi, sqrt, tanh, exp ,sin ,cos
import matplotlib.pyplot as plt
# import scipy.linalg as sla
import time


sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

tau_0 = np.array([[1, 0], [0, 1]])
tau_x = np.array([[0, 1], [1, 0]])
tau_y = np.array([[0, -1j], [1j, 0]])
tau_z = np.array([[1, 0], [0, -1]])

# sigma_0 = 1
# sigma_x = 1
# sigma_y = 1
# sigma_z = 1


size = 1

h = (6.6260693e-034 ) /( 2 *pi)
# hbar
m0 = 9.10938215e-31
# e m
m = h** 2 * 10 ** 37 / (0.27 * 1.6021892 * 2 * m0) * 10 ** 3

# 动能项
theta = 1.2
# twist
am = 0.3691 / (theta * pi / 180)

print("am", am)
Ktheta = 4 * np.pi / (3 * am)

V1 = -30
V11 = 30
eta = 0.15
lam_soc = 30

triangle = kwant.lattice.general([(Ktheta * sqrt(3), 0), (-Ktheta * sqrt(3) / 2, Ktheta * 3 / 2)])


# G1,G2,A,B


def make_system():
    system = kwant.builder.Builder()

    def syst_shape(pos):
        x, y = pos
        return x ** 2 + y ** 2 <= size
        # normest(x1*G1+y1*G2)<= normest(3*G1)

    def onsite(site, kx, ky):
        x, y = site.pos

        return -m * (((kx + x) ** 2 + (ky + y) ** 2) * np.kron(tau_0, sigma_0)
                     + eta * ((kx + x) ** 2 - (ky + y) ** 2) * np.kron(tau_z, sigma_0)
                     + 2 * eta * (kx + x) * (ky + y) * np.kron(tau_x, sigma_0)) + lam_soc / 2 * np.kron(tau_y, sigma_z)

    # 层间耦合，A到B3个方向
    def hopx10(site1, site2):
        x, y = site1.pos
        return np.kron(np.array([[(V1 + V11) / 2, 0], [0, (V1 - V11) / 2]]), sigma_0)

    def hopx01(site1, site2):
        x, y = site1.pos
        return np.kron(np.array([[V1 / 2 - V11 / 4, -np.sqrt(3) / 4 * V11]
                                    , [-np.sqrt(3) / 4 * V11, V1 / 2 + V11 / 4]]), sigma_0)

    def hopx11(site1, site2):
        x, y = site1.pos
        return np.kron(np.array([[V1 / 2 - V11 / 4, np.sqrt(3) / 4 * V11]
                                    , [np.sqrt(3) / 4 * V11, V1 / 2 + V11 / 4]]), sigma_0)

    system[triangle.shape(syst_shape, (0, 0))] = onsite
    system[kwant.builder.HoppingKind((1, 0), triangle, triangle)] = hopx10
    system[kwant.builder.HoppingKind((0, 1), triangle, triangle)] = hopx01
    system[kwant.builder.HoppingKind((1, 1), triangle, triangle)] = hopx11
    # #######这里的a，b是从b到a（site a -b就是从b到a）##########

    return system


def basic_fig():
    gridspec_kw = {'left': 0.16, 'bottom': 0.09, 'right': 0.97, 'top': 0.95}
    fig, ax = plt.subplots(figsize=(2, 1.5), dpi=300, gridspec_kw=gridspec_kw)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['right'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['top'].set_linewidth(0.8)

    return fig,


def blochfunc(wavefunc, G, numband, numsize, r):
    # calculate Bloch function at given point r from the eigenvector of Hamiltonian
    psi_r = np.zeros((numband), dtype='complex128')
    for i in range(numband):
        temp = np.zeros((numsize), dtype='complex128')
        for j in range(numsize):
            temp[j] = np.exp(1j * np.vdot((G[j]), r))  ###γ，M，K，在此更改
        psi_r[i] = np.sum(wavefunc[i, :, 0] * temp)
    return psi_r


def drawfunction(wavefunc, G, numband, numsize):
    xmin = -sqrt(3) * am
    xmax = sqrt(3) * am
    ymin = -2 * am
    ymax = 2 * am
    step = 1
    x = np.arange(xmin, xmax + step, step)
    y = np.arange(ymin, ymax + step, step)
    X, Y = np.meshgrid(x, y)
    psi = np.zeros((numband, len(y), len(x)), dtype='complex128')
    for ii, xx in enumerate(x):
        for jj, yy in enumerate(y):
            psi[:, jj, ii] = blochfunc(wavefunc, G, numband, numsize, np.array([xx, yy]))

    fig, ax = plt.subplots()

    for i in range(numband):
        ax = plt.subplot(1, numband, i + 1)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.xticks(np.arange(-sqrt(3) * am, sqrt(3) * am, sqrt(3) * am / 2),
                   [r'$-\sqrt{3}$', r'$-\sqrt{3}/2$', r'$0$', r'$\sqrt{3}/2$'])
        plt.yticks(np.arange(-2 * am, 2 * am, am), [r'$-2$', r'$-1$', r'$0$', r'$1$'])
        plt.gca().set_aspect('equal', adjustable='box')
        c = ax.contourf(X, Y, abs(psi[i, :, :]) ** 2, cmap='Reds')
        plt.colorbar(c, ax=ax, location='top')


"""Main function"""
start = time.time()

#### 1.0 system
system = make_system()
kwant.plot(system)
system = system.finalized()
print(len(system.sites))

for i in range(len(system.sites)):
    print(i, system.sites[i], system.sites[i].pos)
    print(system.sites[i].tag)
    print("===============================")

N = 301
k = np.zeros((N, 2))

b1 = np.array([Ktheta * sqrt(3), 0])
b2 = np.array([-Ktheta * sqrt(3) / 2, Ktheta * 3 / 2])

kx = 0
ky = 0
x = 1
y = 1
k = np.array([kx, ky])
r = np.array([x, y])
hamiltonian = system.hamiltonian_submatrix(sparse=False, params=dict(kx=kx, ky=ky))
Energies, States = np.linalg.eigh(hamiltonian)
# size_H=len(system.sites)*4
# for i in range(size_H):
#     wave=States[:,i]
G0 = []
G = []
print("len(system.sites)", len(system.sites))
for i in range(len(system.sites)):
    n, m = system.sites[i].tag
    g = n * b1 + m * b2
    G0.append([n, m])
    G.append(g)

numsize = len(G)
band = [numsize - 3, numsize - 2, numsize - 1]
numband = len(band)

wavefunc = np.zeros((numband, numsize, 4), dtype='complex128')
for n in range(numband):
    for i in range(numsize):
        for j in range(4):
            wavefunc[n, i, j] = States[4 * i + j, band[n]]

drawfunction(wavefunc, G, numband, numsize)

end = time.time()
print('Running time: %s seconds' % (end - start))

