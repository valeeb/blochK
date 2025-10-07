#defines a Hamiltonian solely for testing purposes

import numpy as np
from numpy import pi,cos,sin,exp

from blochK.hamiltonian import Hamiltonian2D
import blochK.utils.hamiltonian_fct


def H_AM_fct(kx,ky,t=np.ones((2,2)),t12=0.,mu=-1,m=0): 
    """
    ts: hopping matrix (2x2). axis 0: x,y; axis 1: spin up,down
    t12: hopping between different spins
    """
    Hk = np.zeros((2,2,*kx.shape),dtype=complex) #Basis (up,down)


    #set hamiltonian structure
    Hk[0,0] = -2*t[0,0]*cos(kx) - 2*t[1,0]*cos(ky) - mu - m
    Hk[1,1] = -2*t[0,1]*cos(kx) - 2*t[1,1]*cos(ky) - mu + m
    Hk[0,1] = -2*t12*cos(kx+ky) - 2*t12*cos(kx-ky)

    #make hermitian
    Hk[1,0] = np.conjugate(Hk[0,1])

    return 


def H_2o_AM_fct(kx,ky,t1=1,t2=1,t12=0.,mu=-1,m_F=0,m_AF=0): 
    """
    2 orbitals per spin. Altermagnetic.
    t1: hopping orbital 1 in x direction, orbital 2 in y direction
    t2: hopping orbital 1 in y direction, orbital 2 in x direction
    mu: chemical potential  
    t12: hopping between orbitals
    m_F: Ferro magnetization
    m_AF: Antiferro magnetization
    """
    Hk = np.zeros((4,4,*kx.shape),dtype=complex) #Basis (up,down)

    #set hamiltonian structure
    Hk[0,0] = -2*t1*cos(kx) - 2*t2*cos(ky) - mu
    Hk[1,1] = -2*t2*cos(kx) - 2*t1*cos(ky) - mu
    Hk[0,1] = -2*t12*cos(kx+ky) - 2*t12*cos(kx-ky)

    #make hermitian
    Hk[1,0] = np.conjugate(Hk[0,1])

    #spin degenerate
    Hk[2:,2:] = Hk[:2,:2]

    #add magnetization in z direction
    Hk[0,0] += - m_F - m_AF
    Hk[1,1] += - m_F + m_AF
    Hk[2,2] += + m_F + m_AF
    Hk[3,3] += + m_F - m_AF

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
    Hsquare.add_operator('spinx', np.array([[0,1],[1,0]])) #spin x operator

    return Hsquare


###################################################################################
#Haldane model
###################################################################################

#lattice vectors
n1 = np.array([ 0.5,np.sqrt(3)/2])
n2 = np.array([-0.5,np.sqrt(3)/2])

def Haldane_fct(kx,ky, t=1, t2=0, m=0, mu=0):
    """Defining the Haldane model.
    t: nearest neighbor hopping
    t2: next nearest neighbor hopping (imaginary)
    m: staggered sublattice potential
    mu: chemical potential
    """
    Hk = np.zeros((2,2,*kx.shape), dtype=complex)

    kdotn1 = kx * n1[0] + ky * n1[1]
    kdotn2 = kx * n2[0] + ky * n2[1]
    f = 1 + np.exp(1j*kdotn1) + np.exp(1j*kdotn2)
    g = np.sin(kdotn1) - np.sin(kdotn2) + np.sin(kdotn2 - kdotn1)
    # NNN vectors (same-sublattice)
    b1 = n1 - n2
    b2 = -n1
    b3 = -n2
    kdotb1 = kx * b1[0] + ky * b1[1]
    kdotb2 = kx * b2[0] + ky * b2[1]
    kdotb3 = kx * b3[0] + ky * b3[1]
    dz0 = - mu
    dz  = m + 2.0 * t2 * (np.sin(kdotb1) + np.sin(kdotb2) + np.sin(kdotb3))

    Hk[0,0] = dz0 + dz
    Hk[1,1] = dz0 - dz
    Hk[0,1] = -t * f
    Hk = blochK.hamiltonian_fct.make_hermitian(Hk)
    
    return Hk


def create_Haldane():
    Haldane = Hamiltonian2D(Haldane_fct, n1=n1, n2=n2, basis=['sublattice'],basis_states=['A','B'],param=dict(t2=0.2/(3**0.5)*1.5,m=0.2))
    Haldane.add_operator('sublattice',np.array([1,-1]))

    Haldane.BZ.set_points({
        'K':  (Haldane.BZ.m1 + 2*Haldane.BZ.m2)/3,
        "K'": (2*Haldane.BZ.m1 + Haldane.BZ.m2)/3
    })

    return Haldane