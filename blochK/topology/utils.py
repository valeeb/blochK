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
    y_step = np.einsum('byxi,byxi->byx',np.roll(psis,-1,axis=1),np.roll(np.conjugate(psis),1,axis=1))
    x_step = np.einsum('byxi,byxi->byx',np.roll(psis,-1,axis=2),np.roll(np.conjugate(psis),1,axis=2))
    
    exp_of_flux = np.roll(y_step,1,axis=2) * np.conjugate(np.roll(y_step,-1,axis=2)) * np.roll(x_step,-1,axis=1) * np.conjugate(np.roll(x_step,1,axis=1))
    flux = np.angle(exp_of_flux)/4

    return flux