from blochK.band_sorting import reshape_states
from blochK.hamiltonian_testing import create_Hsquare
from blochK.observable import exp_value_O
import numpy as np


def test_reshape_states():
    """Test the reshape_states function"""
    Hsquare = create_Hsquare()
    ks = Hsquare.BZ.sample(10)
    es,psis = Hsquare.diagonalize(*ks)

    #reshape states based on spin operator
    es_sorted, psis_sorted = reshape_states(es, psis, Hsquare.operator.spin)

    #check shapes
    assert es_sorted.shape[0]==2, "There should be 2 spin sectors"
    assert es_sorted.shape[1]==es.shape[0]//2, "Each sector should have half the number of bands"
    assert psis_sorted.shape[0]==2, "There should be 2 spin sectors"
    assert psis_sorted.shape[1]==psis.shape[0]//2, "Each sector should have half the number of bands"

    #check that the expectation values of the spin operator are sorted correctly
    spin_values = exp_value_O(Hsquare.operator.spin, psis_sorted)

    for i in range(2):
        assert np.all(spin_values[i]==spin_values[i,0,0,0]), "All spin values in a sector should be the same"
