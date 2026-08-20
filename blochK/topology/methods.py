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
        kmesh = Hamiltonian.BZ.sample(Lk,oversample_edge=True)
    else:
        trim_edges = True #kmesh does not correspond to unit cell of reciprocal lattice. Edges are incorrect.
        kmesh = np.array(kmesh)
    
    _,psis = Hamiltonian.diagonalize(*kmesh)

    flux = berry_curvature_state(psis)

    if trim_edges: #for arbitrary kmesh, the edges are not correct.
        return flux[:,1:-1,1:-1], kmesh[:,1:-1,1:-1]
    else: 
        return flux[:,1:-1,1:-1] #edges are also trimmed because they are oversampled
    

def berry_curvature_multiband_state(es,psis,energy=0,project_bands=True):
    """Berry flux of the occupied subspace of a multiband system.

    The routine uses determinant link variables to obtain the trace of the
    non-Abelian Berry curvature.  If the occupation changes across a
    plaquette (as it does at a Fermi surface), the four fixed-occupation
    fluxes associated with its corners are averaged.

    ----------
    Parameters:
    es: ndarray, shape=(n_bands,Lkx,Lky)
        Energies returned from Hamiltonian2D.diagonalize.
    psis: array-like, shape (n_bands,Lkx,Lky,n_orbital)
        Wavefunctions returned from Hamiltonian2D.diagonalize.
    energy: float or one-dimensional array-like
        Energy below which states are occupied.
    project_bands: bool
        If True, discard bands which are never occupied for any requested
        energy. This reduces the overlap-matrix size for models with many
        high-energy bands.
    ----------
    Returns:
        flux : ndarray, shape (n_energy,Lkx,Lky) or (Lkx,Lky)
            Berry flux through each plaquette. A scalar ``energy`` produces
            the two-dimensional result; an array-like produces an explicit
            leading energy axis.

    Notes
    -----
    The mesh is treated as a periodic, rectangular grid.  The value stored at
    ``(x, y)`` is the plaquette whose corners are ``(x, y)``, ``(x-1, y)``,
    ``(x, y-1)``, and ``(x-1, y-1)``.

    The determinant links must be nonzero, and the mesh must be fine enough
    that every relevant plaquette flux lies inside the principal phase branch.
    A singular relevant link raises ``ValueError``; phase-branch convergence
    must be checked by refining the mesh.

    Important:
    Hamiltonian must be formulated in the periodic gauge, Bloch gauge, embedding gauge, periodic Bloch basis or $\vec{k} \cdot \vec{R}_{\alpha}$-gauge (these are all different names for the same gauge) where the phase includes orbital position
    """

    es = np.asarray(es)
    psis = np.asarray(psis)
    if es.ndim != 3:
        raise ValueError("es must have shape (n_bands, Lkx, Lky)")
    if psis.ndim != 4 or psis.shape[:3] != es.shape:
        raise ValueError(
            "psis must have shape (n_bands, Lkx, Lky, n_orbitals) "
            "and match es"
        )

    energy = np.asarray(energy)
    scalar_energy = energy.ndim == 0
    if energy.ndim > 1:
        raise ValueError("energy must be a scalar or a one-dimensional array")
    energy = np.atleast_1d(energy)

    n_energies = energy.size
    Lx, Ly = es.shape[1:]
    if n_energies == 0:
        return np.empty((0, Lx, Ly), dtype=float)

    max_energy = energy.max()

    # project to fewer bands to reduce complexity
    if project_bands:
        occupied_bands = es.min(axis=(1,2))<max_energy
        psis = psis[occupied_bands] #shape = (n_occ,Lkx,Lky,n_orbital)
        es = es[occupied_bands]

    # No occupied states means that the determinant line bundle has rank zero
    # and hence carries zero Berry flux.
    if es.shape[0] == 0:
        flux_exy = np.zeros((n_energies, Lx, Ly), dtype=float)
        return flux_exy[0] if scalar_energy else flux_exy

    # These links are stored at their end point:
    # Mdx[x,y]_mn = <u_n(x-1,y)|u_m(x,y)> (and analogously for y).
    # Transposing the conventional overlap matrix does not change its
    # determinant.
    Mdx_xymn = np.einsum('mxyi,nxyi->xymn',psis,np.roll(np.conjugate(psis),1,axis=1))
    Mdy_xymn = np.einsum('mxyi,nxyi->xymn',psis,np.roll(np.conjugate(psis),1,axis=2))

    #Compute determinant of submatrix. using that det(A.B.C.D) = det(A)*det(B)*det(C)*det(D)
    Udx_xyo = partial_slogdets(Mdx_xymn)
    Udy_xyo = partial_slogdets(Mdy_xymn) 

    # Berry flux through each elementary plaquette for every fixed number of
    # occupied bands.  This phase is already the full one-plaquette flux; the
    # factor 1/4 enters only when the four corner occupations are averaged.
    exp_of_flux_o = Udx_xyo * np.roll(Udy_xyo,1,axis=0) * np.conjugate(np.roll(Udx_xyo,1,axis=1)) * np.conjugate(Udy_xyo) 
    flux_xyo = np.angle(exp_of_flux_o)
    invalid_flux_xyo = (
        (Udx_xyo == 0)
        | (np.roll(Udy_xyo, 1, axis=0) == 0)
        | (np.roll(Udx_xyo, 1, axis=1) == 0)
        | (Udy_xyo == 0)
    )

    #select the flux with the right band multiplicity
    flux_xyo = np.insert(flux_xyo,0,np.zeros_like(flux_xyo[:,:,0]),axis=-1) # add a zero flux layer for the unoccupied bands
    invalid_flux_xyo = np.insert(
        invalid_flux_xyo,
        0,
        np.zeros_like(invalid_flux_xyo[:, :, 0]),
        axis=-1,
    )

    corner_energies = (
        es,
        np.roll(es, 1, axis=1),
        np.roll(es, 1, axis=2),
        np.roll(np.roll(es, 1, axis=1), 1, axis=2),
    )
    x_idx, y_idx = np.indices((Lx, Ly))
    flux_by_corner = []
    for es_corner in corner_energies:
        # Number of occupied bands at this corner for every requested energy.
        multiplicity_exy = (
            es_corner[None] < energy[:, None, None, None]
        ).sum(axis=1)
        selected_invalid = invalid_flux_xyo[
            x_idx[None], y_idx[None], multiplicity_exy
        ]
        if np.any(selected_invalid):
            raise ValueError(
                "an occupied-subspace overlap determinant is zero; "
                "the Berry link is undefined, so refine the k-mesh"
            )
        flux_by_corner.append(
            flux_xyo[x_idx[None], y_idx[None], multiplicity_exy]
        )
    flux_exy = np.mean(flux_by_corner, axis=0)

    return flux_exy[0] if scalar_energy else flux_exy


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
    sigma_xy : float or ndarray
        The anomalous Hall conductivity in units of e^2/h.
    """
    energy = np.asarray(energy)
    scalar_energy = energy.ndim == 0
    if energy.ndim > 1:
        raise ValueError("energy must be a scalar or a one-dimensional array")
    energy = np.atleast_1d(energy)

    kmesh = Hamiltonian.BZ.sample(Lk)
    es,psis = Hamiltonian.diagonalize(*kmesh)

    berry_curv = berry_curvature_multiband_state(es,psis,energy=energy) #shape (energy,Lkx,Lky)

    sigma_xy = np.sum(berry_curv,axis=(-2,-1))/2/pi 

    return sigma_xy[0] if scalar_energy else sigma_xy
