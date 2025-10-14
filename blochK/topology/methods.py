#TODO: add tests for this module

from blochK.hamiltonian import Hamiltonian2D
import numpy as np
from numpy import pi,cos,sin,exp

from .utils import berry_curvature_state, partial_slogdets


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
    

def berry_curvature_multiband_state(es,psis,energy=0,project_bands=True):
    """The total non-Abelian Berry curvature below a given energy of a multiband systems. 
    ----------
    Parameters:
    es: ndarray, shape=(n_bands,Lkx,Lky)
        energys returned from Hamiltonian2D.diagonalize
    psis: array-like, shape (n_bands,Lkx,Lky,n_orbital)
        wavefunction returned from Hamiltonian2D.diagonalize
    energy: float or ndarray
        energy below which the effective total Berry curvature is computed
    project_bands:
        if True the bands are projected to a the subspace where they occupied. Becomes relevant for system with large number of bands above the maximal energy. 
    ----------
    Returns:
        flux : array-like, shape (energy,Lkx,Lky) or (Lkx,Lky)
    """

    #Check if input is correct, make energy an array
    if isinstance(energy,float) or isinstance(energy,int):
        max_energy = energy
        energy = np.array([energy])
    else:
        assert isinstance(energy,np.ndarray), "Energy must be a float,int or a ndarray"
        max_energy = energy.max()

    # project to fewer bands to reduce complexity
    if project_bands:
        occupied_bands = es.min(axis=(1,2))<max_energy
        psis = psis[occupied_bands] #shape = (n_occ,Lkx,Lky,n_orbital)
        es = es[occupied_bands]

    #M(k,dk)_mn = <u_m(kx,ky)|u_n(kx+dkx,ky)> 
    Mdx_xymn = np.einsum('mxyi,nxyi->xymn',psis,np.roll(np.conjugate(psis),1,axis=1))
    Mdy_xymn = np.einsum('mxyi,nxyi->xymn',psis,np.roll(np.conjugate(psis),1,axis=2))

    #Compute determinant of submatrix
    #Udx_xyo = ( det(Mdx_xymn m,n<1), det(Mdx_xymn m,n<2), det(Mdx_xymn m,n<3), ...)
    Udx_xyo = partial_slogdets(Mdx_xymn) 
    Udy_xyo = partial_slogdets(Mdy_xymn) 

    # Flux through plaqeuttes taking into account o bands
    # Udx_o(k) * Udy_o(k+dkx) * Udx_o(k+dky)^* * Udy_o(k)^*
    exp_of_flux_o = Udx_xyo * np.roll(Udy_xyo,1,axis=0) * np.conjugate(np.roll(Udx_xyo,1,axis=1)) * np.conjugate(Udy_xyo) 
    flux_xyo = np.angle(exp_of_flux_o)/4

    #select the flux with the right band multiplicity
    flux_xyo = np.insert(flux_xyo,0,np.zeros_like(flux_xyo[:,:,0]),axis=-1) # add a zero flux layer for the unoccupied bands

    flux_iexy = []
    for es_corners in [es, np.roll(es,-1,axis=1), np.roll(es,-1,axis=2), np.roll(np.roll(es,-1,axis=2),-1,axis=1)]:
        plaqutte_multiplicity_exy = (es_corners[None] < energy[:,None,None,None]).sum(axis=1) #number of occupied bands at each k-point
        #broadcast to the fluxes in different occupation sectors
        n_energys, Lx, Ly = plaqutte_multiplicity_exy.shape
        n_idx, m_idx = np.indices((Lx, Ly))
        n_idx = np.broadcast_to(n_idx, (n_energys, Lx, Ly))
        m_idx = np.broadcast_to(m_idx, (n_energys, Lx, Ly))
        e_idx = np.arange(n_energys)[:, None, None]
        flux_iexy.append(flux_xyo[n_idx, m_idx, plaqutte_multiplicity_exy])
    flux_exy = np.mean(flux_iexy,axis=0)

    if flux_exy.shape[0]==1:
        return flux_exy[0]
    else:
        return flux_exy


def chern_number(Hamiltonian: Hamiltonian2D,Lk=51):
    """Determines the Chern number of each band, all bands must be gapped, i.e. no crossings"""
    A = berry_curvature(Hamiltonian,Lk=Lk)
    return np.sum(A,axis=(1,2))/2/pi


def conductivity_anomalous_Hall(Hamiltonian: Hamiltonian2D,energy=0,Lk=51):
    """
    Computes the intrinsic contribution to the anomalous Hall conductivity at zero temperature.
    Uses the non-Abelian multiband berry curvature.
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

    berry_curv = berry_curvature_multiband_state(es,psis,energy=energy) #shape (energy,Lkx,Lky)

    sigma_xy = np.sum(berry_curv,axis=(-2,-1))/2/pi 
    
    return sigma_xy
