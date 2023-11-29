import numpy as np
import matplotlib.pyplot as plt
import math
import time



chern_change=np.load('chernchange6(60,soc=30,A3=-A).npy')
mu=np.linspace(-60,60,len(chern_change))
plt.plot(mu,chern_change)
# plt.axhline(0.5, color='k', linestyle='--')
plt.xlabel(r'$\mu$')
plt.ylabel("occup states phase")
plt.show()