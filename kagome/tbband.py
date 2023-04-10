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

mu=-1
t=1


def cal1(ks1):
    Eng=[]
    for i in range(len(ks1)):
        kx=ks1[i]
        ky=(1/sqrt(3))*ks1[i]
        k1 = kx
        k2 = kx / 2 + (sqrt(3) / 2) * ky
        k3 = -kx / 2 + (sqrt(3) / 2) * ky
        H_kagome =mu*G[0]-2*t*cos(k1/2)*G[1]-2*t*cos(k2/2)*G[4]-2*t*cos(k3/2)*G[6]
        eng, sta=np.linalg.eigh(H_kagome)
        Eng.append(eng)
        # Eng1=np.array(Eng)
    return Eng

def cal3d(ks1):
    Eng=[]
     for i in range(len(ks1)):
        for j in range(len(ks1)):
            kx=ks1[i]
            ky=ks1[j]
            k1 = kx
            k2 = kx / 2 + (sqrt(3) / 2) * ky
            k3 = -kx / 2 + (sqrt(3) / 2) * ky
            H_kagome =mu*G[0]-2*t*cos(k1/2)*G[1]-2*t*cos(k2/2)*G[4]-2*t*cos(k3/2)*G[6]
            eng, sta=np.linalg.eigh(H_kagome)
            Eng[i,j,:]=eng
     return Eng

kx = np.linspace(-1, 1, 201) * pi
ky=np.linspace(-1, 1, 201) * pi
kx,ky=meshgrid(kx,ky)
# print(cal(ks, 2)[])
# m=[]
# for i in range(len(kx)):
#     m.append(0)

plt.plot(kx, cal1(kx), color='dodgerblue', linewidth=1.0)
# plt.xlim(-2.1,2.1)
# plt.ylim(-1.5,1.5)
plt.xlabel('kx')
plt.ylabel('E')
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(kx, kx,cal3d(kx))

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()


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

