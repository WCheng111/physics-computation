import numpy as np
import matplotlib.pyplot as plt
import math
import time
import multiprocessing
# n = np.load('cherntest1.npy')
# A3 = np.linspace(-4, 4, 150)
# chernone=n[:, 0,0]
# chern2=n[:,0,1]
# chern3=n[:,0,2]
# print(n)
# print(chernone)
# print(chern3)

m=np.load('chernnumber.npy')
A3 = np.linspace(-4, 4, 150)
# print(A3)
chern0=m[:,0,0]
chern1=m[:,0,1]
chern2=m[:,0,2]
plt.plot(A3,chern2,linewidth=3,color='red',label='up band')
plt.plot(A3,chern1,linewidth=3,color='green',label='middle band')
plt.plot(A3,chern0,linewidth=3,color='blue',label='bottom band')
plt.legend(loc='upper center', bbox_to_anchor=(0.2, 0.9))
plt.xlabel(r'$A_{3}^{\prime}$')
plt.ylabel('Chern number')
plt.show()
# print(chern2)