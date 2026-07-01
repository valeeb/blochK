import numpy as np
from blochK.jax.hamiltonian import Hamiltonian2D, BrillouinZone2D
from blochK.jax.hamiltonian import Hamiltonian3D, BrillouinZone3D
from blochK.jax.library_hamiltonians import Hsquare_fct, H3Dsquare_fct


def test_sampleBZ2D():
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


def test_sampleBZ3D():
    Lk = 5
    BZ = BrillouinZone3D()
    ks = BZ.sample(Lk)

    [i, j, k] = np.random.randint(0,Lk-1,size=(3))
    m1_mini = ks[:,i+1,j,k] - ks[:,i,j,k]
    m2_mini = ks[:,i,j+1,k] - ks[:,i,j,k]
    m3_mini = ks[:,i,j,k+1] - ks[:,i,j,k]
    V_mini = np.abs(np.dot(m1_mini, np.cross(m2_mini, m3_mini)))
    assert np.isclose(V_mini*Lk**3, BZ.volume), "Brillouin zone incorrectly sampled"

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(*ks) # original points
    # #ax.scatter(*(ks + BZ.m1[:,None,None,None])) #shifted BZ


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



def test_init_Hamiltonian3D():
    H = Hamiltonian3D(H3Dsquare_fct, n1=np.array([1,0,0]), n2=np.array([0,1,0]), n3=np.array([0,0,1]))  # empty Hamiltonian3D object for testing

    #check brillouin zone correctly defined
    assert np.allclose(H.BZ.m1, 2*np.pi*np.array([1,0,0]))
    assert np.allclose(H.BZ.m2, 2*np.pi*np.array([0,1,0]))
    assert np.allclose(H.BZ.m3, 2*np.pi*np.array([0,0,1]))

    #add operator
    H.add_operator('sz', np.array([[1,0],[0,-1]]))
    assert np.allclose(H.operator.sz, np.array([[1,0],[0,-1]])), "Operator not correctly added"
    H.add_operator('spin', np.array([1,-1]))
    assert np.allclose(H.operator.spin, np.array([1,-1])), "Operator not correctly added"


def test_evaluate_Hamiltonian3D():
    H = Hamiltonian3D(H3Dsquare_fct)
    kx = np.array([1.2])
    ky = np.array([2])
    kz = np.array([3])
    Hk = H.evaluate(kx, ky, kz)
    assert Hk.shape == (H.n_orbitals, H.n_orbitals, *kx.shape), "Hamiltonian not correctly evaluated"


def test_add_Hamiltonian2D():
    H1 = Hamiltonian2D(Hsquare_fct)
    H2 = Hamiltonian2D(Hsquare_fct)

    Hsum = H1 + H2

    kx = np.array([[1.2,0.5],[0.1,0.3]])
    ky = np.array([[2,3],[0.1,0.2]])

    H1k = H1.evaluate(kx, ky)
    H2k = H2.evaluate(kx, ky)
    Hsumk = Hsum.evaluate(kx, ky)

    assert Hsumk.shape == H1k.shape, "Summed Hamiltonian has wrong shape"
    assert np.allclose(Hsumk, H1k + H2k), "Hamiltonian addition does not match elementwise sum"


def test_diagonalize_Hamiltonian3D():
    H = Hamiltonian3D(H3Dsquare_fct)
    kx = np.array([1.2,0.5])
    ky = np.array([2,3])
    kz = np.array([0.1,0.4])
    es, psis = H.diagonalize(kx, ky, kz)
    assert es.shape == (H.n_orbitals, *kx.shape), "Eigenvalues for kx 1D not correctly computed"
    assert psis.shape == (H.n_orbitals, *kx.shape, H.n_orbitals), "Eigenvectors for kx 1D not correctly computed"

    kx = np.array([[1.2,0.5],[0.1,0.3]])
    ky = np.array([[2,3],[0.1,0.2]])
    kz = np.array([[0.4,0.5],[0.6,0.7]])
    es, psis = H.diagonalize(kx, ky, kz)
    assert es.shape == (H.n_orbitals, *kx.shape), "Eigenvalues for kx 2D not correctly computed"
    assert psis.shape == (H.n_orbitals, *kx.shape, H.n_orbitals), "Eigenvectors for kx 2D not correctly computed"