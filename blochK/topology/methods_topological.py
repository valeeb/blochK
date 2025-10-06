#TODO: add tests for this module

from blochK.hamiltonian import Hamiltonian2D
import numpy as np
from numpy import pi,cos,sin,exp

from methods_basic import sample_BZ, sample_reduced_BZ


def Berry_connection(Hamiltonian: Hamiltonian2D, Lk=51,kmesh=None):
    """
    The Berry connection ...
    -----------
    Parameters:
    Hamiltonian : Hamiltonian2D object
    kmesh : array-like, shape (2,Lky,Lkx)
        The k-points where to evaluate the Berry connection, if None the BZ is sampled. Mutually exclusive with Lk
    Lk : int
        linear number of BZ samples. Mutually exclusive with kmesh
    -----------
    Returns:
    A : array-like, shape (nbands,Lky-2,Lkx-2)
    """
    if kmesh is None:
        kmesh = Hamiltonian.BZ.sample(Lk)
    
    es,psis = Hamiltonian.diagonalize(*kmesh)

    #the defined plaquettes includes 4 elementary plaquettes in order to expand around a central k point
    y_step = np.einsum('byxi,byxi->byx',np.roll(psis,-1,axis=1),np.roll(np.conjugate(psis),1,axis=1))
    x_step = np.einsum('byxi,byxi->byx',np.roll(psis,-1,axis=2),np.roll(np.conjugate(psis),1,axis=2))
    
    exp_of_flux = np.roll(y_step,1,axis=2) * np.conjugate(np.roll(y_step,-1,axis=2)) * np.roll(x_step,-1,axis=1) * np.conjugate(np.roll(x_step,1,axis=1))
    flux = np.angle(exp_of_flux)
    
    return flux/4


def Chern_number(Hamiltonian: Hamiltonian2D,Lk=51):
    """Determines the Chern number of each band, all bands must be gapped, i.e. no crossings"""
    A = Berry_connection(Hamiltonian,Lk=Lk)
    return np.sum(A,axis=(1,2))/2/pi