import numpy as np


def berry_curvature_state(psis):
    """Computes the Berry curvature for Bloch wavefunctions
    Parameters:
    psis : array-like, shape (nbands,Lkx,Lky,norb)
        The Bloch wavefunctions at each k-point
    Returns:
        flux : array-like, shape (nbands,Lkx,Lky)
    """
    #the defined plaquettes includes 4 elementary plaquettes in order to expand around a central k point
    x_step = np.einsum('bxyi,bxyi->bxy',np.roll(psis,-1,axis=1),np.roll(np.conjugate(psis),1,axis=1))
    y_step = np.einsum('bxyi,bxyi->bxy',np.roll(psis,-1,axis=2),np.roll(np.conjugate(psis),1,axis=2))
    
    exp_of_flux = np.roll(x_step,-1,axis=2) * np.conjugate(np.roll(x_step,1,axis=2)) * np.roll(y_step,1,axis=1) * np.conjugate(np.roll(y_step,-1,axis=1))
    flux = np.angle(exp_of_flux)/4

    return flux


def partial_dets(T:np.ndarray):
    """Given a tensor of shape T = (... , n, n) compute the determinants of all sub matrices (det T[...,:1,:1], det T[...,:2,:2], det T[...,:3,:3], ...). There might be a faster way because we are double computing all determinants."""
    assert T.shape[-2]==T.shape[-1], "Last 2 axis must have the same length"

    n = T.shape[-1]
    dets = []
    for i in range(1,n+1):
        dets.append(np.linalg.det(T[...,0:i,0:i]))

    return np.moveaxis(dets,0,-1)


def partial_slogdets(T:np.ndarray):
    """Given a tensor of shape T = (... , n, n) compute the slogdeterminants (see numpy documentation) of all sub matrices (det T[...,:1,:1], det T[...,:2,:2], det T[...,:3,:3], ...). There might be a faster way because we are double computing all determinants."""
    assert T.shape[-2]==T.shape[-1], "Last 2 axis must have the same length"

    n = T.shape[-1]
    dets = []
    for i in range(1,n+1):
        dets.append(np.linalg.slogdet(T[...,0:i,0:i])[0])

    return np.moveaxis(dets,0,-1)