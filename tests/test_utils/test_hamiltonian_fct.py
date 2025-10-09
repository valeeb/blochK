import numpy as np
from blochK.utils.hamiltonian_fct import make_hermitian, operator_expand_dims, sx, sy, sz, s0

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

def test_operator_expand_dims():
    kx = np.random.rand(4,2)
    ky = np.random.rand(4,2)

    [sx_exp, sy_exp, sz_exp, s0_exp] = operator_expand_dims([sx,sy,sz,s0], kx)

    assert sx_exp.shape == (2,2,1,1), "operator_expand_dims did not produce the correct shape"
    assert sy_exp.shape == (2,2,1,1), "operator_expand_dims did not produce the correct shape"
    assert sz_exp.shape == (2,2,1,1), "operator_expand_dims did not produce the correct shape"
    assert s0_exp.shape == (2,2,1,1), "operator_expand_dims did not produce the correct shape"


    kx = np.random.rand(3)
    ky = np.random.rand(3)

    [sx_exp, sy_exp, sz_exp, s0_exp] = operator_expand_dims([sx,sy,sz,s0], kx)

    assert sx_exp.shape == (2,2,1), "operator_expand_dims did not produce the correct shape"
    assert sy_exp.shape == (2,2,1), "operator_expand_dims did not produce the correct shape"
    assert sz_exp.shape == (2,2,1), "operator_expand_dims did not produce the correct shape"
    assert s0_exp.shape == (2,2,1), "operator_expand_dims did not produce the correct shape"

    