#defines a Hamiltonian solely for testing purposes

import numpy as np
from numpy import pi,cos,sin,exp

from blochK.hamiltonian import Hamiltonian2D


def Hamiltonian_func0(kx,ky,t=1,mu=-1): 
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

def Hsquare_fct(kx,ky,t=1,mu=-1,m=0): 
    """
    t: NN hopping 
    mu: chemical potential
    m: FM
    """
    Hk = np.zeros((2,2,*kx.shape),dtype=complex) #Basis (up,down)

    #set hamiltonian structure
    Hk[0,0] = -2*t*cos(kx) - 2*t*cos(ky) - mu

    #make hermitian
    Hk[1,0] = np.conjugate(Hk[0,1])

    #spin degenerate
    Hk[1:,1:] = Hk[:1,:1]

    #add magnetization in z direction
    Hk[0,0] -= m
    Hk[1,1] += m

    return Hk


def create_Hsquare():
    """Create Hsquare function with default parameters"""
    n1 = np.array([1,0])
    n2 = np.array([0,1])
    Hsquare = Hamiltonian2D(Hsquare_fct, basis_states=['up','down'], basis=['spin'], n1=n1, n2=n2)
    Hsquare.add_operator('spin', np.array([1,-1])) #diagnonal part of sz

    return Hsquare

