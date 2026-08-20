from blochK.topology import berry_curvature,chern_number, conductivity_anomalous_Hall, berry_curvature_multiband_state
from blochK.hamiltonian_testing import create_Haldane
import numpy as np


def test_berry_curvature():
    Haldane = create_Haldane()
    Lk = 11
    ks = np.meshgrid(np.linspace(-4.6,4.6,Lk),np.linspace(-4.6,4.6,Lk),indexing='ij')

    #compute Berry curvature on a 50x50 k-mesh
    Omega,kmesh = berry_curvature(Haldane, kmesh=ks)

    #check that the Berry curvature is antisymmetric in kx and ky
    assert Omega.shape == (2,Lk-2,Lk-2), "For Haldane model, there are two bands, edges are trimmed"
    assert kmesh.shape == (2,Lk-2,Lk-2), "edges are trimmed"
    

def test_chern_number():
    Haldane = create_Haldane()

    #compute Chern number
    C = chern_number(Haldane,Lk=21)

    #check that the Chern number is close to 1 and -1 for the two bands
    assert np.allclose(np.abs(C), [1,1]), "Chern number should be close to 1 and -1 for the two bands"


def test_berry_curvature_multiband_state():
    Haldane = create_Haldane()
    es,psis = Haldane.diagonalize(*Haldane.BZ.sample(11))

    assert berry_curvature_multiband_state(es,psis,energy=0).shape == (11,11)

    energy = np.array([-1,0])
    assert berry_curvature_multiband_state(es,psis,energy=energy).shape == (2,11,11)

    single_energy = np.array([0])
    assert berry_curvature_multiband_state(es,psis,energy=single_energy).shape == (1,11,11)


def test_berry_curvature_multiband_state_chern_number():
    """A fixed occupied subspace must give its complete, unscaled flux."""
    Haldane = create_Haldane()
    Lk = 21
    es, psis = Haldane.diagonalize(*Haldane.BZ.sample(Lk))

    flux = berry_curvature_multiband_state(es, psis, energy=0)
    occupied_chern = flux.sum() / (2 * np.pi)

    assert np.allclose(occupied_chern, chern_number(Haldane, Lk=Lk)[0])


def test_berry_curvature_multiband_state_two_occupied_bands():
    """The determinant link adds the curvature of an occupied multiplet."""
    Haldane = create_Haldane()
    # The composite flux must remain inside the principal phase branch on
    # every plaquette. An 11x11 mesh is too coarse after doubling the Chern
    # sector and aliases one plaquette phase by 2*pi.
    es, psis = Haldane.diagonalize(*Haldane.BZ.sample(21))
    Lx, Ly = es.shape[1:]

    # Two identical, mutually orthogonal copies of the Haldane model. The
    # first two states are occupied and each contributes the same Chern number.
    es4 = np.stack((es[0], es[0], es[1], es[1]))
    psis4 = np.zeros((4, Lx, Ly, 4), dtype=complex)
    psis4[0, ..., :2] = psis[0]
    psis4[1, ..., 2:] = psis[0]
    psis4[2, ..., :2] = psis[1]
    psis4[3, ..., 2:] = psis[1]

    flux = berry_curvature_multiband_state(es4, psis4, energy=0)
    occupied_chern = flux.sum() / (2 * np.pi)
    assert np.allclose(np.abs(occupied_chern), 2)

    # A k-dependent U(2) rotation changes the eigenvector frame, but not the
    # occupied subspace or its determinant Wilson loop.
    kx, ky = np.meshgrid(
        np.linspace(0, 2 * np.pi, Lx, endpoint=False),
        np.linspace(0, 2 * np.pi, Ly, endpoint=False),
        indexing="ij",
    )
    theta = 0.37 * np.sin(kx) + 0.21 * np.cos(ky)
    phase = np.exp(1j * (0.43 * np.cos(kx) - 0.31 * np.sin(ky)))
    rotation = np.empty((Lx, Ly, 2, 2), dtype=complex)
    rotation[..., 0, 0] = np.cos(theta)
    rotation[..., 0, 1] = phase * np.sin(theta)
    rotation[..., 1, 0] = -np.conjugate(phase) * np.sin(theta)
    rotation[..., 1, 1] = np.cos(theta)

    rotated_psis = psis4.copy()
    rotated_psis[:2] = np.einsum(
        "xyab,bxyi->axyi", rotation, psis4[:2]
    )
    rotated_flux = berry_curvature_multiband_state(
        es4, rotated_psis, energy=0
    )
    assert np.allclose(rotated_flux, flux)


def test_berry_curvature_multiband_state_uses_matching_corners():
    Haldane = create_Haldane()
    _, psis = Haldane.diagonalize(*Haldane.BZ.sample(11))

    # Use one geometrical band and prescribe a nonuniform occupation. For one
    # band, the metallic prescription is exactly the full flux multiplied by
    # the fraction of occupied corners of the same plaquette.
    es = np.ones((1, 11, 11))
    es[0, 3, 4] = -1
    psis = psis[:1]
    full_flux = berry_curvature_multiband_state(es, psis, energy=2)
    partial_flux = berry_curvature_multiband_state(es, psis, energy=0)

    occupied = (es[0] < 0).astype(int)
    occupied_corner_fraction = (
        occupied
        + np.roll(occupied, 1, axis=0)
        + np.roll(occupied, 1, axis=1)
        + np.roll(np.roll(occupied, 1, axis=0), 1, axis=1)
    ) / 4
    assert np.allclose(partial_flux, full_flux * occupied_corner_fraction)


def test_berry_curvature_multiband_state_empty_subspace():
    Haldane = create_Haldane()
    es, psis = Haldane.diagonalize(*Haldane.BZ.sample(11))
    energy = es.min() - 1

    flux = berry_curvature_multiband_state(es, psis, energy=energy)

    assert flux.shape == (11, 11)
    assert np.all(flux == 0)


def test_berry_curvature_multiband_state_rejects_singular_links():
    es = -np.ones((1, 3, 3))
    psis = np.zeros((1, 3, 3, 2), dtype=complex)
    psis[0, 0, :, 0] = 1
    psis[0, 1, :, 1] = 1
    psis[0, 2, :, 0] = 1

    try:
        berry_curvature_multiband_state(es, psis, energy=0)
    except ValueError as error:
        assert "overlap determinant is zero" in str(error)
    else:
        raise AssertionError("a singular occupied-subspace link must fail")



def test_conductivity_anomalous_Hall():
    Haldane = create_Haldane()
    Lk = 11

    #trivial phase
    Haldane.update_params(dict(t2=-0.2/(3**0.5)*0.5,m=0.2))
    cond_trivial = conductivity_anomalous_Hall(Haldane,energy=np.array([0,1]),Lk=Lk)
    assert cond_trivial.shape == (2,), "There are two energies, should return two conductivities"

    cond_single_array = conductivity_anomalous_Hall(Haldane, energy=np.array([0]), Lk=Lk)
    cond_scalar = conductivity_anomalous_Hall(Haldane, energy=0, Lk=Lk)
    assert cond_single_array.shape == (1,)
    assert np.ndim(cond_scalar) == 0
