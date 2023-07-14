import numpy as np
import math
import matplotlib.pyplot as plt


def E(x):
    E=x+2
    return E
x=np.linspace(-2,2,4)
Eng=[]
for i in range(len(x)):
    Eng.append(E(i))
plt.plot(x,Eng)
plt.show()
print(x)