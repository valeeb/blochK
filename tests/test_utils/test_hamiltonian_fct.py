import pytest
import numpy as np
from blochK.utils.hamiltonian_fct import make_hermitian, operator_expand_dims, kron_first_axes, sx, sy, sz, s0

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


def test_kron_first_axes_shape():
    """Check that only the first two axes are expanded."""
    A = np.ones((2, 3))
    B = np.ones((4, 5, 6, 7))

    C = kron_first_axes(A, B)

    assert C.shape == (8, 15, 6, 7)


def test_kron_first_axes_values():
    """Check numerical equivalence with applying np.kron slice by slice."""
    A = np.arange(6).reshape(2, 3)
    B = np.arange(4 * 5 * 2).reshape(4, 5, 2)

    C = kron_first_axes(A, B)

    # Reference implementation:
    # apply np.kron(A, B[..., i]) for every remaining index
    C_ref = np.empty((2 * 4, 3 * 5, 2), dtype=np.result_type(A, B))

    for i in range(B.shape[2]):
        C_ref[..., i] = np.kron(A, B[..., i])

    np.testing.assert_array_equal(C, C_ref)


def test_kron_first_axes_complex():
    """Check that complex-valued arrays are handled correctly."""
    A = np.array([[1 + 1j, 2], [3, 4j]])
    B = np.ones((2, 2, 3), dtype=complex)

    C = kron_first_axes(A, B)

    assert C.shape == (4, 4, 3)

    for i in range(B.shape[2]):
        np.testing.assert_array_equal(C[..., i], np.kron(A, B[..., i]))


def test_kron_first_axes_invalid_input():
    """Check that invalid dimensionalities raise errors."""
    A = np.ones((2, 2, 2))
    B = np.ones((2, 2))

    with pytest.raises(ValueError):
        kron_first_axes(A, B)

    A = np.ones((2, 2))
    B = np.ones((2,))

    with pytest.raises(ValueError):
        kron_first_axes(A, B)