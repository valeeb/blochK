#How to use conserved quantities to relabel and reshape eigenstates

from blochK.observable import exp_value_O
import numpy as np

def reshape_states(es, psis, operator):
    """
    Relabel and reshape eigenstates based on conserved quantities. (band n)->(conserved quantity q, band n')
    Parameters:
    es : np.ndarray
        The eigenvalues array of shape (N_bands, ...).
    psis : np.ndarray
        The eigenstates array of shape (N_bands, ..., N_states).
    operator : np.ndarray
        The operator matrix used for sorting. Must commute with the Hamiltonian. Must have integer eigenvalues. Assumes that each sector has the same number of bands.
    Returns:
    es_sorted : np.ndarray
        The sorted eigenvalues array of shape (N_q_values, N_bands_per_q, ...).
    psis_sorted : np.ndarray
        The sorted eigenstates array of shape (N_q_values, N_bands_per_q, ..., N_states).
    """
    if operator.ndim==1:
        qs = operator
    elif operator.ndim==2:
        qs = np.linalg.eigvalsh(operator)
    else:
        raise ValueError("Operator must be either 1D or 2D array.")
    numb_sectors = len(np.unique(qs))
    q_values = exp_value_O(operator, psis)

    #sort by conserved quantity
    asort = np.argsort(q_values,axis=0)
    es_sorted = np.take_along_axis(es, asort,axis=0)
    psis_sorted = np.take_along_axis(psis, asort[...,np.newaxis],axis=0)

    #reshape into sectors. conserved quantity sectors are now the first axis
    es_sorted = es_sorted.reshape((numb_sectors, -1) + es_sorted.shape[1:])    
    psis_sorted = psis_sorted.reshape((numb_sectors, -1) + psis_sorted.shape[1:])

    #sort within each sector by energy
    asort_energy = np.argsort(es_sorted,axis=1) 
    es_sorted = np.take_along_axis(es_sorted, asort_energy,axis=1)
    psis_sorted = np.take_along_axis(psis_sorted, asort_energy[...,np.newaxis],axis=1) 

    return es_sorted, psis_sorted