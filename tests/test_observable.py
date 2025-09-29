from blochK.hamiltonian_testing import create_Hsquare
import blochK.observable as observable
import numpy as np


def test_exp_value_O():
    param = dict(t=1)
    kx = np.array([0, 2, 1])
    ky = np.array([1, 1, 0])
    H = create_Hsquare()
    es, psi = H.diagonalize(kx, ky,param)

    #call O1 check shape
    O1 = observable.exp_value_O(H.operator.spin, psi)
    assert O1.shape == (2, 3)

    #call O2 check shape
    O2 = observable.exp_value_O(np.diag(H.operator.spin), psi)
    assert O2.shape == (2, 2, 3)

    #check if quadratic and linear version of exp_value_O give the same result
    assert np.allclose(O1, np.diagonal(np.real(O2)).T)


def test_isDegenerateIn():
    param = dict(t=1)
    kx = np.array([0, 2, 1])
    ky = np.array([1, 1, 0])
    H = create_Hsquare()
    es, psi = H.diagonalize(kx, ky,param)

    O = observable.exp_value_O(H.operator.spin, psi)

    deg = observable.isDegenerateIn(es, O)
    assert deg.shape == es.shape
    assert np.all(deg == True)



#############################################################################################################################################
#conductivity
#############################################################################################################################################


def test_conductivity():
    param = dict(t=1,m=0.4)
    H = create_Hsquare()
    H.set_params(param)

    sigma = observable.conductivity(H)
    assert sigma.shape == (2, 2)
    assert np.isclose(sigma[0,0], sigma[1,1]) #xx and yy should be equal by symmetry

    sigma = observable.conductivity(H,operator=H.operator.spin)
    assert sigma.shape == (2, 2)
    assert np.isclose(sigma[0,0], sigma[1,1]) #xx and yy should be equal by symmetry


def test_conductivity_orbital_resolved():
    H = create_Hsquare()
    H.set_params(dict(t=1,m=0.4))

    sigma = observable.conductivity_orbital_resolved(H)
    assert sigma.shape == (2, 2, 2)


def test_conductivity_and_conductivity_orbital_resolved():
    H = create_Hsquare()
    H.set_params(dict(t=1,m=0.4))
    
    projector = np.array([1,0]) #projector on first orbital
    sigma1 = observable.conductivity(H, operator=projector)
    sigma_or = observable.conductivity_orbital_resolved(H)

    #sum over orbital-resolved conductivity to get total conductivity
    sigma2 = np.sum(sigma_or*projector[:,np.newaxis,np.newaxis], axis=0)

    assert np.isclose(sigma1, sigma2).all()


#############################################################################################################################################
#QPIs
#############################################################################################################################################
def test_conductivity():
    param = dict(t=1,m=0.4)
    H = create_Hsquare()
    H.set_params(param)

    ldos = observable.local_dos_QPI(H,Gamma=0.2,Lk=5)
    assert ldos.shape == (5, 5)

    ldos = observable.local_dos_QPI(H,Gamma=0.2,Lk=5,operator=0.2*H.operator.spinx)
    assert ldos.shape == (5, 5)


#############################################################################################################################################
#MLD
#############################################################################################################################################
def test_magnetic_linear_dichroism():
    H = create_Hsquare()
    H.set_params(dict(t=1,m=0.4))

    omegas = np.linspace(-3,3,5)
    mld = observable.magnetic_linear_dichroism(H,omegas)

    assert mld.shape == (len(omegas),), "MLD should have the same length as omegas"
    assert np.isclose(mld, 0).all(), "MLD should be zero for block diagonal matrix"


#############################################################################################################################################
#helper functions
#############################################################################################################################################

def test_find_Gamma():
    H = create_Hsquare()
    H.set_params(dict(t=1,mu=-1,m=0.3))

    Lk = 10
    ks = H.BZ.sample(Lk)
    es,_ = H.diagonalize(*ks)

    Gamma = observable.find_Gamma(es)
    print('Using Gamma = ',Gamma)   