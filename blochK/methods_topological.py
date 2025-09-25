#TODO: add tests for this module


import numpy as np
from numpy import pi,cos,sin,exp

from methods_basic import sample_BZ, sample_reduced_BZ


def Berry_connection(Hamiltonian_fct,Hparam={},Lq=51):
    """Determines the Berry connection in the reduced BZ for given Hamiltonian_fct"""
    ks = sample_reducedBZ(Lq=Lq)
    #ks = sample_BZ(Lq=Lq)
    
    es,psis = eigs_H(*ks,Hamiltonian_fct,Hparam)
    
    #the defined plaquettes includes 4 elementary plaquettes in order to expand around a central k point
    y_step = np.einsum('byxi,byxi->byx',np.roll(psis,-1,axis=1),np.roll(np.conjugate(psis),1,axis=1))
    x_step = np.einsum('byxi,byxi->byx',np.roll(psis,-1,axis=2),np.roll(np.conjugate(psis),1,axis=2))
    
    exp_of_flux = np.roll(y_step,1,axis=2) * np.conjugate(np.roll(y_step,-1,axis=2)) * np.roll(x_step,-1,axis=1) * np.conjugate(np.roll(x_step,1,axis=1))
    flux = np.angle(exp_of_flux)
    
    return flux/4


def Chern_number(Hamiltonian_fct,Hparam={},Lq=51):
    """Determines the Chern number of each band, all bands must be gapped, i.e. no crossings"""
    A = Berry_connection(Hamiltonian_fct,Hparam=Hparam,Lq=Lq)
    return np.sum(A,axis=(1,2))/2/pi