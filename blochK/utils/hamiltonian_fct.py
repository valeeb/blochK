import numpy as np

#Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
s0 = np.array([[1, 0], [0, 1]], dtype=complex)


def make_hermitian(Hk:np.ndarray):
    """Adds the lower diagonal to the upper diagonal to make a matrix hermitian. Ignores the diagonal.
    Parameters:
    Hk : array-like, shape (n,n,...)
        The input matrix, where the first two indices are the matrix indices
    """
    assert Hk.ndim>=2, "Hk must be at least 2-dimensional"
    assert Hk.shape[0]==Hk.shape[1], "Hk must be a square matrix in its first two indices"

    # Hk.shape (n1,n2,..)
    Hk = np.moveaxis(np.moveaxis(Hk, 0, -1), 0, -1) #shape (...,n1,n2)

    #select upper triangular part
    Hk_tri = np.triu(Hk, 1) #shape (...,n1,n2)
    Hk_diag = Hk-np.triu(Hk, 1) - np.tril(Hk, -1) #shape (...,n1,n2)


    H_hermitian = Hk_tri + np.conjugate(np.swapaxes(Hk_tri, -2, -1)) + Hk_diag #shape (...,n1,n2)
    H_hermitian = np.moveaxis(np.moveaxis(H_hermitian, -1, 0), -1, 0) #shape (n1,n2,...)

    return H_hermitian


def operator_expand_dims(list_of_operators,momenta):
    """
    Expands the dimensions of a list of operators such that they can be multiplied with momenta arrays.
    -------------
    Parameters:
    list_of_operators : list of np.ndarray
        List of operators to be expanded. Each operator should be a 2D array of shape (n,n).
    momenta : np.ndarray. Input of a Hamiltonian function. 
    -------------
    Returns:
    list of np.ndarray
        List of operators with expanded dimensions. Each operator will have shape (n,n,1,1,...,1) where the number of 1's is equal to the number of dimensions in momenta.
    -------------
    Example:
    Hk = np.zeros((2,2,*kx.shape),dtype=complex)
    [s0,sx] = operator_expand_dims([s0,sx], kx)
    Hk += s0*mu + sx*2*t*(np.cos(kx)+np.cos(ky))
    """
    return [np.expand_dims(op,axis=tuple([-i for i in range(1,len(momenta.shape)+1)])) for op in list_of_operators]
