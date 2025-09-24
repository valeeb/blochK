from blochK.hamiltonian_testing import Spin_operator,Hamiltonian0
import blochK.observable as observable
import numpy as np

def test_Hamiltonian0():
    param = dict(t=1)
    kx = np.array([0, 2])
    ky = np.array([1, 1])
    Hk = Hamiltonian0(kx, ky,**param)

    assert Hk.shape == (2, 2, 2)


def test_eigs_H():
    param = dict(t=1)
    kx = np.array([0, 2, 1])
    ky = np.array([1, 1, 0])
    es, psi = observable.eigs_H(kx, ky, Hamiltonian0, param)

    assert es.shape == (2, 3)
    assert psi.shape == (2, 3, 2)


def test_exp_value_O():
    param = dict(t=1)
    kx = np.array([0, 2, 1])
    ky = np.array([1, 1, 0])
    es, psi = observable.eigs_H(kx, ky, Hamiltonian0, param)

    #call O1 check shape
    O1 = observable.exp_value_O(Spin_operator, psi)
    assert O1.shape == (2, 3)

    #call O2 check shape
    O2 = observable.exp_value_O(np.diag(Spin_operator), psi)
    assert O2.shape == (2, 2, 3)

    #check if quadratic and linear version of exp_value_O give the same result
    assert np.allclose(O1, np.diagonal(np.real(O2)).T)


def test_isDegenerateIn():
    param = dict(t=1)
    kx = np.array([0, 2, 1])
    ky = np.array([1, 1, 0])
    es, psi = observable.eigs_H(kx, ky, Hamiltonian0, param)

    O = observable.exp_value_O(Spin_operator, psi)
    
    deg = observable.isDegenerateIn(es, O) 
    assert deg.shape == es.shape
    assert np.all(deg == True)


def test_conductivity():
    param = dict(t=1)
    sigma = observable.conductivity(Hamiltonian_fct=Hamiltonian0, Hparam=param)
    assert sigma.shape == (2, 2)
    assert np.isclose(sigma[0,0], sigma[1,1])


def test_conductivity_orbital_resolved():
    param = dict(t=1)
    sigma = observable.conductivity_orbital_resolved(Hamiltonian_fct=Hamiltonian0, Hparam=param)
    assert sigma.shape == (2, 2, 2)


def test_conductivity_and_conductivity_orbital_resolved():
    param = dict(t=1)
    projector = np.array([1,0]) #projector on first orbital
    sigma1 = observable.conductivity(Hamiltonian_fct=Hamiltonian0, Hparam=param, operator=projector)
    sigma_or = observable.conductivity_orbital_resolved(Hamiltonian_fct=Hamiltonian0, Hparam=param)

    #sum over orbital-resolved conductivity to get total conductivity
    sigma2 = np.sum(sigma_or*projector[:,np.newaxis,np.newaxis], axis=0)

    assert np.isclose(sigma1, sigma2).all()
