import kwant
import  numpy as np
import matplotlib
from  matplotlib import pyplot as plt
import math
import mathematical

sigma = list(range(4))
sigma[0] = np.array([[1, 0], [0, 1]])
sigma[1] = np.array([[0, 1], [1, 0]])
sigma[2] = np.array([[0, -1j], [1j, 0]])
sigma[3] = np.array([[1, 0], [0, -1]])

gama=[[0 for i in range(4)] for j in range(4)]

for i in range(4):
    for j in range(4):
        gama[i][j] = np.kron(sigma[i], sigma[j])


Lx=10
Ly=10
Lz=10

M0=-1
M1=1
M2=1
A1=1
A2=1
A3=1

def make_system_TI():
        a=10**-3
        def system_shape(pos):
            x,y=pos
            return (0<x<Lx and 0<y<Ly )
        def one_site(site,kz):
            z=site.pos
            #H_TI=(M0+2M1(1-cos(kx))+2M1(1-cos(ky))+2M2(1-cos(kz)))gama[0][3]
                #  +A1sin(kx)gama[2][1]+A2sin(ky)gama[2][2]+A3sin(kz)gammap[2][3]
            H= (M0 + 2 * M2 * (1 - math.cos(kz))) * gama[0][3] + A3 * math.sin(kz)
            return H
        def hooping_x(site):
            hoppingx=