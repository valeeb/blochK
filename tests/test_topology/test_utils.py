import numpy as np
from blochK.topology.utils import partial_dets,partial_slogdets


def test_partial_dets():
    shape = (2,3,4,4)
    T = np.random.rand(*shape) + 1j*np.random.rand(*shape)
    dets = partial_dets(T)

    assert dets.shape == T.shape[:-1], "Shape missmatch between input and output"


def test_partial_slogdets():
    shape = (2,3,4,4)
    T = np.random.rand(*shape) + 1j*np.random.rand(*shape)
    dets = partial_slogdets(T)

    assert dets.shape == T.shape[:-1], "Shape missmatch between input and output"


def test_partial_slogdets_vs_dets():
    shape = (2,3,4,4)
    T = np.random.rand(*shape) + 1j*np.random.rand(*shape)
    x0 = np.angle(partial_slogdets(T))
    x1 = np.angle(partial_dets(T))

    assert np.allclose(x0,x1), "partial slogdets and dets do not agree"