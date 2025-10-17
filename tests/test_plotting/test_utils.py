from blochK.plotting.utils import path, sample_square, sample_reduced_square, extent
import pytest
import numpy as np


def test_path_basic():
    points = np.array([[0,0],[1,0],[1,1]])
    pathx, pathk, pathxPoints = path(points, N_samples=10)
    
    # Check types
    assert isinstance(pathx, np.ndarray)
    assert isinstance(pathk, np.ndarray)
    assert isinstance(pathxPoints, np.ndarray)
    
    # Check shapes
    assert pathx.shape[0] == pathk.shape[0]
    assert pathk.shape[1] == points.shape[1]
    assert pathxPoints.shape[0] == len(points)


def test_sample_square_basic():
    Lq = 5
    ks = sample_square(Lq)
    
    # Check type and shape
    assert isinstance(ks, np.ndarray)
    assert ks.shape == (2, Lq, Lq)  # 2D square sampling


def test_sample_reduced_square_basic():
    Lq = 6
    ks = sample_reduced_square(Lq)
    
    # Check type and shape
    assert isinstance(ks, np.ndarray)
    assert ks.shape[0] == 2  # 2D coordinates
    assert ks.shape[1] == Lq
    assert ks.shape[2] == Lq


def test_extent():
    # 1) Integer input
    ext = extent(4)
    assert isinstance(ext, list)
    assert len(ext) == 4
    assert ext[0] < ext[1] and ext[2] < ext[3]
    
    # 2) Meshgrid input
    Lk = 4
    kx, ky = np.meshgrid(np.linspace(-1, 1, Lk), np.linspace(-1, 1, Lk), indexing='ij')
    ext = extent([kx, ky])
    assert isinstance(ext, list)
    assert len(ext) == 4
    # Check that extent slightly extends beyond the min/max
    assert ext[0] < kx.min() and ext[1] > kx.max()
    assert ext[2] < ky.min() and ext[3] > ky.max()
    
    # 3) Invalid input
    with pytest.raises(ValueError):
        extent(3.5)
    
    with pytest.raises(ValueError):
        extent("invalid")
