import numpy as np
import sympy
import qsymm

#空间反演
Inverse=np.array([[-1,0,0,0,0,0],[0,-1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],
                  [0,0,0,0,0,1]])
IS = qsymm.inversion(3, U=Inverse)


R4z=np.array([[1-1j,0,0,0,0,0],[0,1+1j,0,0,0,0],[0,0,1-1j,0,0,0],
              [0,0,0,1+1j,0,0],[0,0,0,0,-1-1j,0],[0,0,0,0,0,-1+1j]])
rot4z = qsymm.rotation(1/4,axis=[0,0,1],U=R4z)



R2x=np.array([[0,1j,0,0,0,0],[1j,0,0,0,0,0],[0,0,0,1j,0,0],[0,0,1j,0,0,0],[0,0,0,0,0,1j],
                  [0,0,0,0,1j,0]])

rot2x= qsymm.rotation(1/2, axis=[1,0,0],U=R2x)


T=np.array([[0,-1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,-1,0,0,0],[0,0,0,0,0,-1],
                  [0,0,0,0,1,0]])
TS = qsymm.time_reversal(3, U=T)

symmetries = [rot4z, rot2x,TS,IS]

dim = 3
total_power = 2

family = qsymm.continuum_hamiltonian(symmetries, dim, total_power, prettify=True)
qsymm.display_family(family)