import numpy as np
import matplotlib.pyplot as plt
import math
import time



chern_change=np.load('Berryphase(A3=-A,soc=0.01).npy')
mu=np.linspace(-60,60,len(chern_change))
plt.scatter(mu,chern_change,s=2)
plt.axhline(0.5, color='k', linestyle='--')
plt.axhline(-0.5, color='k', linestyle='--')
plt.xlabel(r'$\mu$')
plt.ylabel("occup states phase")
plt.show()