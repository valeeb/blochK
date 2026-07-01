#defines a Hamiltonian solely for testing purposes

import numpy as np
import jax.numpy as jnp

from blochK.jax.hamiltonian import Hamiltonian2D



def H_AM_fct(kx,ky,t=None,t12=0.,mu=-1,m=0): 
    """
    ts: hopping matrix (2x2). axis 0: x,y; axis 1: spin up,down
    t12: hopping between different spins
    """
    kx = jnp.asarray(kx)
    ky = jnp.asarray(ky)
    t = jnp.ones((2,2)) if t is None else jnp.asarray(t)
    Hk = jnp.zeros((2,2,*kx.shape),dtype=jnp.complex128) #Basis (up,down)


    #set hamiltonian structure
    Hk = Hk.at[0,0].set(-2*t[0,0]*jnp.cos(kx) - 2*t[1,0]*jnp.cos(ky) - mu - m)
    Hk = Hk.at[1,1].set(-2*t[0,1]*jnp.cos(kx) - 2*t[1,1]*jnp.cos(ky) - mu + m)
    Hk = Hk.at[0,1].set(-2*t12*jnp.cos(kx+ky) - 2*t12*jnp.cos(kx-ky))

    #make hermitian
    Hk = Hk.at[1,0].set(jnp.conjugate(Hk[0,1]))

    return Hk


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
    kx = jnp.asarray(kx)
    ky = jnp.asarray(ky)
    Hk = jnp.zeros((4,4,*kx.shape),dtype=jnp.complex128) #Basis (up,down)

    #set hamiltonian structure
    Hk = Hk.at[0,0].set(-2*t1*jnp.cos(kx) - 2*t2*jnp.cos(ky) - mu)
    Hk = Hk.at[1,1].set(-2*t2*jnp.cos(kx) - 2*t1*jnp.cos(ky) - mu)
    Hk = Hk.at[0,1].set(-2*t12*jnp.cos(kx+ky) - 2*t12*jnp.cos(kx-ky))

    #make hermitian
    Hk = Hk.at[1,0].set(jnp.conjugate(Hk[0,1]))

    #spin degenerate
    Hk = Hk.at[2:,2:].set(Hk[:2,:2])

    #add magnetization in z direction
    Hk = Hk.at[0,0].add(- m_F - m_AF)
    Hk = Hk.at[1,1].add(- m_F + m_AF)
    Hk = Hk.at[2,2].add(+ m_F + m_AF)
    Hk = Hk.at[3,3].add(+ m_F - m_AF)

    return Hk


def Hsquare_fct(kx,ky,t=1,mu=-1,m=0): 
    """
    Simplest 2D square lattice model with ferromagnetism.
    t: NN hopping 
    mu: chemical potential
    m: FM
    """
    kx = jnp.asarray(kx)
    ky = jnp.asarray(ky)
    Hk = jnp.zeros((2,2,*kx.shape),dtype=jnp.complex128) #Basis (up,down)

    #set hamiltonian structure
    Hk = Hk.at[0,0].set(-2*t*jnp.cos(kx) - 2*t*jnp.cos(ky) - mu)

    #make hermitian
    Hk = Hk.at[1,0].set(jnp.conjugate(Hk[0,1]))

    #spin degenerate
    Hk = Hk.at[1:,1:].set(Hk[:1,:1])

    #add magnetization in z direction
    Hk = Hk.at[0,0].add(-m)
    Hk = Hk.at[1,1].add(+m)

    return Hk


def H3Dsquare_fct(kx,ky,kz,t=1,mu=-1,m=0): 
    """
    Simplest 3D square lattice model with ferromagnetism.
    t: NN hopping 
    mu: chemical potential
    m: FM
    """
    kx = jnp.asarray(kx)
    ky = jnp.asarray(ky)
    kz = jnp.asarray(kz)
    Hk = jnp.zeros((2,2,*kx.shape),dtype=jnp.complex128) #Basis (up,down)

    #set hamiltonian structure
    Hk = Hk.at[0,0].set(-2*t*jnp.cos(kx) - 2*t*jnp.cos(ky) - 2*t*jnp.cos(kz) - mu)

    #make hermitian
    Hk = Hk.at[1,0].set(jnp.conjugate(Hk[0,1]))

    #spin degenerate
    Hk = Hk.at[1:,1:].set(Hk[:1,:1])

    #add magnetization in z direction
    Hk = Hk.at[0,0].add(-m)
    Hk = Hk.at[1,1].add(+m)

    return Hk


def H_pAFM_diag_fct(kx,ky,t=1,delta=0.,mu=-1,m=0): 
    """
    Minimal model spin-diagonal p-wave AFM. 
    t: NN hopping
    delta: unisotropy in p-wave pairing
    """
    kx = jnp.asarray(kx)
    ky = jnp.asarray(ky)
    Hk = jnp.zeros((2,2,*kx.shape),dtype=jnp.complex128) #Basis (up,down)
    #set hamiltonian structure
    Hk = Hk.at[0,0].set(-2*(t+delta)*jnp.cos(kx) - 2*t*jnp.cos(ky) - mu + delta * (jnp.sin(kx) + jnp.sin(2*kx + delta)))
    Hk = Hk.at[1,1].set(-2*(t+delta)*jnp.cos(kx) - 2*t*jnp.cos(ky) - mu - delta * (jnp.sin(kx) + jnp.sin(2*kx - delta)))

    return Hk


def H_pAFM_fct(kx,ky,t=1,alpha=None,mu=-1,m=0): 
    """
    Minimal model for a p-wave AFM. 
    t: NN hopping
    d_phase: positive phase shift in x direction of the up spin and phase shift in -x direction of the down spin
    """
    kx = jnp.asarray(kx)
    ky = jnp.asarray(ky)
    alpha = jnp.zeros(2) if alpha is None else jnp.asarray(alpha)
    Hk = jnp.zeros((4,4,*kx.shape),dtype=jnp.complex128) #Basis (up A, up B, down A, down B)

    #set hamiltonian structure
    Hk = Hk.at[0,0].set(-2*t*jnp.cos(kx) - 2*t*jnp.cos(ky) + alpha[0]*jnp.sin(kx) + alpha[1]*jnp.sin(ky) - mu)
    Hk = Hk.at[1,1].set(-2*t*jnp.cos(kx) - 2*t*jnp.cos(ky) + alpha[0]*jnp.sin(kx) + alpha[1]*jnp.sin(ky)- mu)
    Hk = Hk.at[2,2].set(-2*t*jnp.cos(kx) - 2*t*jnp.cos(ky) - alpha[0]*jnp.sin(kx) - alpha[1]*jnp.sin(ky)- mu)
    Hk = Hk.at[3,3].set(-2*t*jnp.cos(kx) - 2*t*jnp.cos(ky) - alpha[0]*jnp.sin(kx) - alpha[1]*jnp.sin(ky)- mu)
    #make hermitian
    Hk = Hk.at[0,2].set(-m)
    Hk = Hk.at[2,0].set(-m)
    Hk = Hk.at[1,3].set(+m)
    Hk = Hk.at[3,1].set(+m)

    return Hk


def create_Hsquare():
    """Create Hsquare function with default parameters"""
    n1 = jnp.array([1,0])
    n2 = jnp.array([0,1])
    Hsquare = Hamiltonian2D(Hsquare_fct, basis_states=['up','down'], basis=['spin'], n1=n1, n2=n2)
    Hsquare.add_operator('spin', jnp.array([1,-1])) #diagnonal part of sz
    Hsquare.add_operator('spinx', jnp.array([[0,1],[1,0]])) #spin x operator

    return Hsquare


###################################################################################
#Haldane model
###################################################################################

#lattice vectors
n1 = jnp.array([ 0.5,jnp.sqrt(3)/2])
n2 = jnp.array([-0.5,jnp.sqrt(3)/2])

def Haldane_fct(kx,ky, t=1, t2=0, m=0, mu=0):
    """Defining the Haldane model.
    t: nearest neighbor hopping
    t2: next nearest neighbor hopping (imaginary)
    m: staggered sublattice potential
    mu: chemical potential
    """
    kx = jnp.asarray(kx)
    ky = jnp.asarray(ky)
    Hk = jnp.zeros((2,2,*kx.shape), dtype=jnp.complex128)

    kdotn1 = kx * n1[0] + ky * n1[1]
    kdotn2 = kx * n2[0] + ky * n2[1]
    f = 1 + jnp.exp(1j*kdotn1) + jnp.exp(1j*kdotn2)
    g = jnp.sin(kdotn1) - jnp.sin(kdotn2) + jnp.sin(kdotn2 - kdotn1)
    # NNN vectors (same-sublattice)
    b1 = n1 - n2
    b2 = -n1
    b3 = -n2
    kdotb1 = kx * b1[0] + ky * b1[1]
    kdotb2 = kx * b2[0] + ky * b2[1]
    kdotb3 = kx * b3[0] + ky * b3[1]
    dz0 = - mu
    dz  = m + 2.0 * t2 * (jnp.sin(kdotb1) + jnp.sin(kdotb2) + jnp.sin(kdotb3))

    Hk = Hk.at[0,0].set(dz0 + dz)
    Hk = Hk.at[1,1].set(dz0 - dz)
    Hk = Hk.at[0,1].set(-t * f)
    Hk = Hk.at[1,0].set(jnp.conjugate(Hk[0,1]))
    
    return Hk


def create_Haldane():
    Haldane = Hamiltonian2D(Haldane_fct, n1=n1, n2=n2, basis=['sublattice'],basis_states=['A','B'],param=dict(t2=0.2/(3**0.5)*1.5,m=0.2))
    Haldane.add_operator('sublattice',jnp.array([1,-1]))

    Haldane.BZ.set_points({
        'K':  (Haldane.BZ.m1 + 2*Haldane.BZ.m2)/3,
        "K'": (2*Haldane.BZ.m1 + Haldane.BZ.m2)/3
    })

    return Haldane