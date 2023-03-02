import kwant
from matplotlib import pyplot
import tinyarray
import numpy as np
import time
from numpy import exp, pi, kron, cos, sin, sqrt, pi, cosh, tanh
from pathos.multiprocessing import Pool
from scipy import sparse, integrate
from matplotlib import pyplot as plt

sigma0 = np.array([[1, 0], [0, 1]])
sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])

tau0 = sigma0
taux = sigmax
tauy = sigmay
tauz = sigmaz

s0 = sigma0
sx = sigmax
sy = sigmay
sz = sigmaz

Lz = 16
Lx = 10
Ly = 50

normal = 4
Lsc = (Lz - normal) // 2

M = -1.5
A1 = 2
A2 = 2
tx = 1
ty = 1
tz = 1
delta = 0.5
mu = 0.

pen_len = Lz  # 100
decay_l = 0.2
C0 = ((pen_len + (Lz // 2)) / decay_l) / tanh((Lz // 2) / decay_l)


def kwant_cubic_conner_TSC_Magnetic_Y(theta):
    def sys_shape(pos):
        es = 1e-3
        x, y, z = pos
        return (-Lx // 2 - es < x < Lx // 2 + es and -Ly // 2 - es < y < Ly // 2 + es \
                    and -es - Lz // 2 < z < Lz // 2 + es)

    def onsite(site,phi_sc):
        x, y, z = site.pos
        if z >= (Lz // 2 - Lsc):
            h_sc = kron(taux, kron(s0, sigma0))
        elif z <= -(Lz // 2 - Lsc):
            h_sc = cos(phi_sc) * kron(taux, kron(s0, sigma0)) + sin(phi_sc) * kron(tauy, kron(s0, sigma0))
        else:
            h_sc = 0 * kron(taux, kron(s0, sigma0))
        onsite_e = ((M + 2 * tx + 2 * ty + 2 * tz) * kron(s0, sigmaz) - mu * kron(s0, sigma0))


        return kron(tauz, onsite_e) + delta * h_sc

    def hopping_y(site1, site2, eta):
        x1, y1, z1 = site1.pos
        x2, y2, z2 = site2.pos
        z = (z1 + z2) / 2

        hopy_e = (-ty * kron(s0, sigmaz) + A2 / (2j) * kron(sy, sigmax))

        phase = eta * cos(theta) * (-z * pen_len / (Lz // 2) - C0 * (1 / cosh(z / decay_l) ** 2) * tanh(z / decay_l) * \
                                    ((1 / 3) * (y2 ** 3 - y1 ** 3)) / decay_l)

        return kron(tauz, hopy_e) * cos(2 * pi * phase) + 1j * kron(tau0, hopy_e) * sin(2 * pi * phase)

    def hopping_x(site1, site2, eta):
        x1, y1, z1 = site1.pos
        x2, y2, z2 = site2.pos
        z = (z1 + z2) / 2
        hopx_e = (-tx * kron(s0, sigmaz) + A2 / (2j) * kron(sx, sigmax))
        phase = eta * sin(theta) * (z * pen_len / (Lz // 2) + C0 * (1 / cosh(z / decay_l) ** 2) * tanh(z / decay_l) *\
                                    ((1 / 3) * (x2 ** 3 - x1 ** 3)) / decay_l)

        return kron(tauz, hopx_e) * cos(2 * pi * phase) + 1j * kron(tau0, hopx_e) * sin(2 * pi * phase)

    def hopping_z(site1, site2, eta):
        x1, y1, z1 = site1.pos
        x2, y2, z2 = site2.pos
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2

        hopz_e = -tz * kron(s0, sigmaz) + A1 / (2j) * kron(sz, sigmax)
        phase = eta * (sin(theta) * x - cos(theta) * y) * C0 * decay_l * (tanh(z2 / decay_l) - tanh(z1 / decay_l))

        return kron(tauz, hopz_e) * cos(2 * pi * phase) + 1j * kron(tau0, hopz_e) * sin(2 * pi * phase)

    lat = kwant.lattice.cubic(1, norbs=8)
    sys = kwant.Builder()
    sys[lat.shape(sys_shape, (1, 1, 1))] = onsite
    sys[kwant.builder.HoppingKind((1, 0, 0), lat, lat)] = hopping_x
    sys[kwant.builder.HoppingKind((0, 1, 0), lat, lat)] = hopping_y
    sys[kwant.builder.HoppingKind((0, 0, 1), lat, lat)] = hopping_z

    return sys

Phi_sc=1
del_phi = np.pi
eta = del_phi / (2 * pi * 2 * (Ly// 2) * (Lz + 2 * pen_len))

B_theta=np.linspace(0,1,21)*pi
density_point = []
for j in range(len(B_theta)):

    system = kwant_cubic_conner_TSC_Magnetic_Y(j).finalized()
    # kwant.plot(system,site_size=0.2,site_lw=0.001,hop_lw=0.1)
    ham_mat = system.hamiltonian_submatrix(sparse=True, params=dict(eta=eta, phi_sc=Phi_sc * pi))
    energies, states = sparse.linalg.eigsh(ham_mat, k=8, sigma=0, return_eigenvectors=True, which='LM')
    rho = kwant.operator.Density(system)
    density = rho(states[:, 0]) + rho(states[:, 1])
    density = density/ np.max(density)


    for i in range(len(system.site)):
        if np.abs(system.site[i].pos[0]-Lx/2)<0.1 and np.abs(system.site[i].pos[1])<0.1 and\
            np.abs(system.site[i].pos[2])<0.1:
            density_point.append(density[i])
plt.plot(B_theta,density_point)

