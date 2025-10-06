import numpy as np
from blochK.utils.hamiltonian_fct import make_hermitian

def test_make_hermitian():
    Hk = np.zeros(shape=(3,3,4,2),dtype=complex)

    Hk[0,0] = np.random.rand(4,2)
    Hk[1,1] = np.random.rand(4,2)
    Hk[2,2] = np.random.rand(4,2)
    Hk[0,1] = np.random.rand(4,2) + 1j*np.random.rand(4,2)
    Hk[0,2] = np.random.rand(4,2) + 1j*np.random.rand(4,2)
    Hk[1,2] = np.random.rand(4,2) + 1j*np.random.rand(4,2)

    Hk_herm = make_hermitian(Hk)
    assert np.allclose(Hk_herm, np.conjugate(np.swapaxes(Hk_herm,0,1))), "make_hermitian did not produce a hermitian matrix"