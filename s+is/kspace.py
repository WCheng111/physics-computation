import numpy as np
import matplotlib.pyplot as plt
import kwant
import tinyarray
import math

m=1
Uhh=1
Uhe=1
Delta1h_int=1
Delta2h_int=1
Delta3e_int=1
Delta_int=1
def H(k,Delta1h,Delta2h,Delta3e):
    H=np.array([
        [k**2/(2*m),-Uhe*Delta2h-2*Uhe*Delta3e,0,0,0,0,0,0],
        [-Uhh*np.conj(Delta2h)-2*Uhe*np.conj(Delta3e),-k**2/(2*m),0,0,0,0,0,0],
        [0,0,k**2/(2*m),-Uhh*Delta1h-2*Uhe*Delta3e,0,0,0,0],
        [0,0,Uhe*(np.conj(Delta1h)+np.conj(Delta2h)),-k**2/(2*m),0,0,0,0],
        [0,0,0,0,-k**2/(2*m),Uhe*(Delta1h+Delta2h),0,0],
        [0,0,0,0,Uhe*(np.conj(Delta1h)+np.conj(Delta2h)),k**2/(2*m),0,0],
        [0,0,0,0,0,0,-k**2/(2*m),Uhe*(Delta1h+Delta2h)],
        [0, 0, 0, 0, 0, 0,Uhe * (np.conj(Delta1h) + np.conj(Delta2h)), k**2/(2*m)]
    ])
    return H
def H1(k,Delta):
    H=np.array([[k**2/(2*m),Delta],[np.conj(Delta),-k**2/(2*m)]])
    return H

k=np.linspace(0,800,1000)
#零温单带迭代

def calate_2(k1,Delta):
    Delta2 = 0
    for i in range(len(k1)):
        eng,sta=np.linalg.eigh(H1(k1[i],Delta))
        sta0=sta[:,0]
        sta1=sta[:,1]
        sta00=sta0[0]*sta0[1]
        sta10=sta1[0]*sta1[1]
        Delta2+=sta00-np.conj(sta10)
    return Delta2
print(calate_2(k,1))
print(H1(2,1))
# def calate_3(k1):
#     Delta2 = 0
#     for i in range(len(k1)):
#         Delta2+=2
#     return Delta2
# print(calate_3(k))
Delta_change=2

Delta_final=1
while Delta_change>1e-10:
    Delta_change=abs(Delta_final-calate_2(k,Delta_final))
    Delta_final=calate_2(k,Delta_final)
    print(Delta_final)


# print(np.conj(2+1j))