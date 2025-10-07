#TODO: add tests for this module

from blochK.hamiltonian import Hamiltonian2D
import numpy as np
from numpy import pi,cos,sin,exp

from .utils import berry_curvature_state


def berry_curvature(Hamiltonian: Hamiltonian2D, Lk=51,kmesh=None):
    """
    The Berry curvature 
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

    flux = berry_curvature_state(psis)

    if trim_edges: #for arbitrary kmesh, the edges are not correct.
        return flux[:,1:-1,1:-1], kmesh[:,1:-1,1:-1]
    else:
        return flux


def chern_number(Hamiltonian: Hamiltonian2D,Lk=51):
    """Determines the Chern number of each band, all bands must be gapped, i.e. no crossings"""
    A = berry_curvature(Hamiltonian,Lk=Lk)
    return np.sum(A,axis=(1,2))/2/pi


def conductivity_anomalous_Hall(Hamiltonian: Hamiltonian2D,energy=0,Lk=51):
    """
    Computes the intrinsic contribution to the anomalous Hall conductivity at zero temperature.
    -----------
    Parameters:
    Hamiltonian : Hamiltonian2D object
    energy : float or ndarray 
    Lk : int
    -----------
    Returns:
    sigma_xy : float
        The anomalous Hall conductivity in units of e^2/hbar
    """
    if isinstance(energy,float) or isinstance(energy,int):
        energy = np.array([energy])
    else:
        assert energy.ndim==1, "energy must be a scalar or a 1D array"

    kmesh = Hamiltonian.BZ.sample(Lk)
    es,psis = Hamiltonian.diagonalize(*kmesh)

    berry_curv = berry_curvature_state(psis)[None,:,:,:] #shape (energy,nbands,Lkx,Lky)
    fermi_occup = es[None,:,:,:]<energy[:,None,None,None]
    sigma_xy = np.sum(berry_curv*fermi_occup,axis=(1,2,3))/2/pi 
    
    return sigma_xy
