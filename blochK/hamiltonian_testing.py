#defines a Hamiltonian solely for testing purposes

import numpy as np
from numpy import pi,cos,sin,exp

#operators
Spin_operator = np.array([1,-1]) #spin up +1, spin down -1

#Definitions for the Square lattice
#lattice vectors
n1 = np.array([1,0])
n2 = np.array([0,1])

# Area of unit cell (2D cross product)
A = n1[0]*n2[1] - n1[1]*n2[0]
#reciprocal lattice vectors
m1 = 2*np.pi/A * np.array([n2[1], -n2[0]])
m2 = 2*np.pi/A * np.array([-n1[1], n1[0]])

# Define High symmetry points in BZ
points_BZ = {
    "\Gamma": [0,0],
    "X": [1,0],
    "Y": [0,1],
    "R": [1,1],
    "R'": [1,-1],
    "-R": [-1,1],
    "-R'": [-1,-1]
}


def Hamiltonian0(kx,ky,t=1,mu=-1): 
    """
    t: NN hopping 
    """
    Hk = np.zeros((2,2,*kx.shape),dtype=complex) #Basis (up,down)

    #set hamiltonian structure
    Hk[0,0] = -2*t*cos(kx) - 2*t*cos(ky) - mu

    #make hermitian
    Hk[1,0] = np.conjugate(Hk[0,1])

    #spin degenerate
    Hk[1:,1:] = Hk[:1,:1]

    return Hk

