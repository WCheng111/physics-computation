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


P = list(range(4))
P[0] = np.array([[1, 0], [0, 1]])
P[1] = np.array([[0, 1], [1, 0]])
P[2] = np.array([[0, -1j], [1j, 0]])
P[3] = np.array([[1, 0], [0, -1]])


Pauli = [[0 for i in range(4)] for j in range(4)]


for i in range(4):
    for j in range(4):
        Pauli[i][j] = np.kron(P[i], P[j])


M00 = -1
M01 = 1
M02 = 1
A1 = 0.5
A2 =0.5
A3= 0.5

socm1 = np.array([[-1, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0], [0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])


def cal_E(ks1):
    Eng=[]
    for i in range(len(ks1)):
        kx = ks1[0]
        ky=0
        kz=0
        H_TI= (M00 + 2 * M01 * (1 - cos(kx)) + 2 * M01 * (1 - cos(ky)) + 2 * M02 * (1 - cos(kz))) * Pauli[0][3] + A1*/
            /sin(kx)*Pauli[2][1]+A2*sin(ky)*Pauli[2][2]+A3*sin(kz)*Pauli[2][3]
    return np.