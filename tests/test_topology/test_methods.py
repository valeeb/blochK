from blochK.topology import berry_curvature,chern_number
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