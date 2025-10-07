#TODO: add tests for this module

from blochK.hamiltonian import Hamiltonian2D
import numpy as np
from numpy import pi,cos,sin,exp


def berry_curvature(Hamiltonian: Hamiltonian2D, Lk=51,kmesh=None):
    """
    The Berry curvature ...
    -----------
    Parameters:
    Hamiltonian : Hamiltonian2D object
    kmesh : array-like, shape (2,Lky,Lkx)
        The k-points where to evaluate the Berry curvature, if None the BZ is sampled. Mutually exclusive with Lk
    Lk : int
        linear number of BZ samples. Mutually exclusive with kmesh
    -----------
    Returns:
    flux : array-like, shape (nbands,Lkx,Lky) if kmesh is None or (nbands,Lkx-2,Lky-2) if kmesh was provided
        The Berry curvature of each band at each k-point
    (kmesh): array-like, shape (Lkx-2,Lky-2) if kmesh was provided
    """
    if kmesh is None:
        trim_edges = False
        kmesh = Hamiltonian.BZ.sample(Lk)
    else:
        trim_edges = True #kmesh does not correspond to unit cell of reciprocal lattice. Edges are incorrect.
        kmesh = np.array(kmesh)
    
    _,psis = Hamiltonian.diagonalize(*kmesh)

    #the defined plaquettes includes 4 elementary plaquettes in order to expand around a central k point
    y_step = np.einsum('byxi,byxi->byx',np.roll(psis,-1,axis=1),np.roll(np.conjugate(psis),1,axis=1))
    x_step = np.einsum('byxi,byxi->byx',np.roll(psis,-1,axis=2),np.roll(np.conjugate(psis),1,axis=2))
    
    exp_of_flux = np.roll(y_step,1,axis=2) * np.conjugate(np.roll(y_step,-1,axis=2)) * np.roll(x_step,-1,axis=1) * np.conjugate(np.roll(x_step,1,axis=1))
    flux = np.angle(exp_of_flux)/4

    if trim_edges: #for arbitrary kmesh, the edges are not correct.
        return flux[:,1:-1,1:-1], kmesh[:,1:-1,1:-1]
    else:
        return flux


def chern_number(Hamiltonian: Hamiltonian2D,Lk=51):
    """Determines the Chern number of each band, all bands must be gapped, i.e. no crossings"""
    A = berry_curvature(Hamiltonian,Lk=Lk)
    return np.sum(A,axis=(1,2))/2/pi