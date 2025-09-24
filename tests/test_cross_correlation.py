import time as time
import numpy as np
from blochK.utils.cross_correlation import cross_correlation, cross_correlation_broadcast, cross_correlation_2loops, cross_correlation_4loops

def test_cross_correlation():
    """Tests the 4 different implementations of the cross correlation function"""
    Lq1=5
    Lq2=3
    ndimH = 2

    Greensfct_yxab = np.random.rand(Lq1,Lq2,ndimH,ndimH)*10 + 10j*np.random.rand(Lq1,Lq2,ndimH,ndimH)


    t0 = time.time()
    z0 = cross_correlation_4loops(Greensfct_yxab)
    t1 = time.time()
    print('time for 4 loop version ', t1-t0)
    t0 = time.time()
    z1 = cross_correlation_2loops(Greensfct_yxab)
    t1 = time.time()
    print('time for 2 loop version ', t1-t0)
    t0 = time.time()
    z2 = cross_correlation_broadcast(Greensfct_yxab)
    t1 = time.time()
    print('time for broadcast version ', t1-t0)
    t0 = time.time()
    z3 = cross_correlation(Greensfct_yxab)
    t1 = time.time()
    print('time for FFT version ', t1-t0)

    assert np.allclose(z0, z1), "cross_correlation_2loops not equal to cross_correlation_4loops"
    assert np.allclose(z0, z2), "cross_correlation_broadcast not equal to cross_correlation_4loops"
    assert np.allclose(z0, z3), "cross_correlation (FFT) not equal to cross_correlation_4loops"
    assert np.allclose(z1, z2), "cross_correlation_broadcast not equal to cross_correlation_2loops"
    assert np.allclose(z1, z3), "cross_correlation (FFT) not equal to cross_correlation_2loops"
    assert np.allclose(z2, z3), "cross_correlation (FFT) not equal to cross_correlation_broadcast"