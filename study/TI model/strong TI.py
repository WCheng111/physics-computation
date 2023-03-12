import kwant
import  numpy as np
import matplotlib
from  matplotlib import pyplot as plt


sigma = list(range(4))
sigma[0] = np.array([[1, 0], [0, 1]])
sigma[1] = np.array([[0, 1], [1, 0]])
sigma[2] = np.array([[0, -1j], [1j, 0]])
sigma[3] = np.array([[1, 0], [0, -1]])

gama=[[0 for i in range(4)] for j in range(4)]

for i in range(4):
    for j in range(4):
        gama[i][j] = np.kron(sigma[i], sigma[j])

def TI1():
