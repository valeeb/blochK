import numpy as np
from .utils import d_matrix


# TODO: Add a test for this function
def two_band_hamiltonian_2d(
    kx: np.ndarray, ky: np.ndarray, t_values: dict
) -> np.ndarray:

    hk = np.zeros((2, 2, *kx.shape), dtype=complex)
    # add each set of hoppings
    for hopping in t_values:

        d_mat = d_matrix(t_values[hopping])  # hopping matrix

        # the (0,0) component has no hermitian conjugate
        # so must be real and will be doubled later, let's half it
        if hopping == (0, 0):
            d_mat = d_mat / 2

        phases = np.exp(-1j * kx * hopping[0] - 1j * ky * hopping[1])

        term_to_add = np.multiply.outer(d_mat, phases)
        hk += term_to_add

    return hk + np.moveaxis(hk, 1, 0).conj()


# TODO: Add a test for this function
def two_band_hamiltonian_3d(
    kx: np.ndarray, ky: np.ndarray, t_values: dict, len_z=20, open_bcs_z=False
) -> np.ndarray:

    hk = np.zeros((2 * len_z, 2 * len_z, *kx.shape), dtype=complex)

    # add each set of hoppings
    for hopping in t_values:

        d_mat = d_matrix(t_values[hopping])  # hopping matrix

        phases = np.exp(-1j * kx * hopping[0] - 1j * ky * hopping[1])  # phases
        terms_to_add = np.einsum("ij, ... -> ij...", d_mat, phases)

        ind1 = np.arange(len_z)
        ind2 = ind1 + hopping[2]

        if open_bcs_z:  # either assign only hoppings that don't cross the boundary
            ind1 = ind1[(ind2 < len_z) * (ind2 >= 0)]
            ind2 = ind2[(ind2 < len_z) * (ind2 >= 0)]
        else:  # or wrap around the periodic boundaries
            ind2 = ind2 % len_z

        hk[ind1 * 2, ind2 * 2] += terms_to_add[0, 0]
        hk[ind2 * 2, ind1 * 2] += terms_to_add[0, 0].conj()

        hk[ind1 * 2+1 , ind2 * 2] += terms_to_add[1, 0]
        hk[ind2 * 2, ind1 * 2+1] += terms_to_add[1, 0].conj()

        hk[ind1 * 2, ind2 * 2+1] += terms_to_add[0, 1]
        hk[ind2 * 2+1, ind1 * 2] += terms_to_add[0, 1].conj()

        hk[ind1 * 2+1, ind2 * 2+1] += terms_to_add[1, 1]
        hk[ind2 * 2+1, ind1 * 2+1] += terms_to_add[1, 1].conj()



    return hk





def d_vector(t_values, k_vals):
    """Computes the d-vector at each k-point given a dictionary of hopping terms, works in arbitrary
    dimensions (usually 2D or 3D).
    Args:
        t_values (dict): a dictionary of every lattice vector and the associated (complex)
            Pauli vector for it
        k_vals (np.ndarray): array of k-values (shape (D, N...)), with D the dimension and N
            the number of k-points in many axes as it could be an array
    Returns:
        np.ndarray: d-vectors at each k-point
    """

    d_vec = np.zeros((4, *k_vals.shape[1:]))

    for hopping in t_values:

        phase = np.exp(1j * np.einsum("i,i...->...", np.array(hopping), k_vals))
        v = np.array(t_values[hopping])
        values = np.einsum("j,k...->jk...", v, phase)
        d_vec += values.real * 2

    return d_vec

