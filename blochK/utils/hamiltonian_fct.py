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


def kron_first_axes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute the Kronecker product of a 2D array with the first two axes
    of an N-dimensional array.

    The operation is equivalent to applying np.kron(A, B) to the first
    two dimensions of B while leaving all remaining dimensions unchanged.

    Parameters
    ----------
    A : np.ndarray, shape (a1, a2)
        Two-dimensional input array.

    B : np.ndarray, shape (b1, b2, b3, ..., bn)
        N-dimensional input array. The Kronecker product is applied only
        to the first two axes.

    Returns
    -------
    C : np.ndarray, shape (a1*b1, a2*b2, b3, ..., bn)
        Resulting array with the first two axes combined according to the
        Kronecker product and all remaining axes preserved.

    Examples
    --------
    >>> A = np.arange(6).reshape(2, 3)
    >>> B = np.arange(24).reshape(2, 4, 3)
    >>> C = kron_first_axes(A, B)
    >>> C.shape
    (4, 12, 3)

    Notes
    -----
    The implementation uses np.einsum to avoid explicitly constructing
    large intermediate Kronecker matrices.
    """
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    if B.ndim < 2:
        raise ValueError("B must have at least 2 dimensions")

    a1, a2 = A.shape
    b1, b2 = B.shape[:2]

    C = np.einsum("ij,kl...->ikjl...", A, B)

    return C.reshape(a1 * b1, a2 * b2, *B.shape[2:])