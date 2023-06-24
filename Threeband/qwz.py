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
s0 = np.eye(2)
sx = np.array([[0,1], [1,0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1,0],[0,-1]])

def qwz(a=1, m=-1.6, L=20):  # a = 1为晶格常数, L, W 为长宽

    lat = kwant.lattice.square(a)  # 创建晶格，方格子

    syst = kwant.Builder(kwant.TranslationalSymmetry([0, a]))  # 建立中心体系

    # 中心区
    syst[(lat(x, 0) for x in range(L))] = m * sz  # onsite能是矩阵也可以被Builder识别
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = 0.5 * (sz - 1j * sx)  #
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = 0.5 * (sz - 1j * sy)

    kwant.plot(syst, dpi=300)  # 把电极-中心区-电极图画出来，通过图像可以看出有没有写错
    syst = syst.finalized()  # 结束体系的制作。这个语句不可以省。这个语句是把Builder对象转换成可计算的对象。
    return syst
kwant.plotter.bands(qwz(), momenta=np.linspace(-pi, pi, 201), show=False, dpi=600)
plt.xlabel("momentum [(lattice constant)$^{-1}$]")
plt.ylabel("Energy [$t$]")
plt.show()