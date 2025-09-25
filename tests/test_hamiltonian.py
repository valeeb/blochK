import numpy as np
from blochK.hamiltonian import Hamiltonian2D, BrillouinZone2D
from blochK.hamiltonian_testing import Hsquare_fct


def test_sampleBZ():
    Lk = 5
    BZ = BrillouinZone2D()
    ks = BZ.sample(Lk)

    [i, j] = np.random.randint(0,Lk-1,size=(2))
    m1_mini = ks[:,i+1,j] - ks[:,i,j]
    m2_mini = ks[:,i,j+1] - ks[:,i,j]
    A_mini = np.abs(m1_mini[0]*m2_mini[1] - m1_mini[1]*m2_mini[0])
    assert np.isclose(A_mini*Lk**2, BZ.area), "Brillouin zone incorrectly sampled"

    # plt.scatter(*ks)
    # plt.scatter(ks[0]+BZ.m1[0],ks[1]+BZ.m1[1])
    # plt.scatter(ks[0]+BZ.m2[0],ks[1]+BZ.m2[1])


def test_init_Hamiltonian2D():
    H = Hamiltonian2D(Hsquare_fct, n1=np.array([1,0]), n2=np.array([0,1]))  # empty Hamiltonian2D object for testing

    #check brillouin zone correctly defined
    assert np.allclose(H.BZ.m1, 2*np.pi*np.array([1,0]))
    assert np.allclose(H.BZ.m2, 2*np.pi*np.array([0,1]))

    #add operator
    H.add_operator('sz', np.array([[1,0],[0,-1]]))
    assert np.allclose(H.operator.sz, np.array([[1,0],[0,-1]])), "Operator not correctly added"
    H.add_operator('spin', np.array([1,-1]))
    assert np.allclose(H.operator.spin, np.array([1,-1])), "Operator not correctly added"


def test_evaluate_Hamiltonian2D():
    H = Hamiltonian2D(Hsquare_fct)
    kx = np.array([1.2])
    ky = np.array([2])
    Hk = H.evaluate(kx, ky)
    assert Hk.shape == (H.n_orbitals, H.n_orbitals, *kx.shape), "Hamiltonian not correctly evaluated"


def test_diagonalize_Hamiltonian2D():
    H = Hamiltonian2D(Hsquare_fct)
    kx = np.array([1.2,0.5])
    ky = np.array([2,3])
    es, psis = H.diagonalize(kx, ky)
    assert es.shape == (H.n_orbitals, *kx.shape), "Eigenvalues for kx 1D not correctly computed"
    assert psis.shape == (H.n_orbitals, *kx.shape, H.n_orbitals), "Eigenvectors for kx 1D not correctly computed"

    kx = np.array([[1.2,0.5],[0.1,0.3]])
    ky = np.array([[2,3],[0.1,0.2]])
    es, psis = H.diagonalize(kx, ky)
    assert es.shape == (H.n_orbitals, *kx.shape), "Eigenvalues for kx 2D not correctly computed"
    assert psis.shape == (H.n_orbitals, *kx.shape, H.n_orbitals), "Eigenvectors for kx 2D not correctly computed"