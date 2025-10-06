import numpy as np

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