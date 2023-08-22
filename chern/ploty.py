import matplotlib.pyplot as plt
import numpy as np

chern=np.load('chernnum.npy')
plotx=np.linspace(-45,45,200)
plt.scatter(plotx,chern[:,0],s=1)
plt.scatter(plotx,chern[:,1],s=1)
plt.scatter(plotx,chern[:,2],s=1)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'Chern number')
plt.show()