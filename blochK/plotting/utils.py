import numpy as np
from numpy import pi,cos,sin,exp


def path(points,N_samples=100):
    """Construct path along points (at least 2 points needed).

    Returns:
    pathx: shape=(N_samples) variable parametrizing the path,
    pathk: shape=(N_samples,dim) k values for each path variable,
    pathxPoints: position of labels"""
    points = np.array(points)
    pathk=[]; pathx=[];pathxPoints=[]
    ts = np.linspace(0,1,N_samples,endpoint=False)
    norm0=0
    for i in range(1,len(points)):
        pathxPoints.append(norm0)
        if i==len(points)-1:
            ts = np.linspace(0,1,N_samples+1,endpoint=True)
        vec = points[i]-points[i-1]
        pathk.append(np.outer(ts,vec)+points[i-1])
        pathx.append(ts*np.linalg.norm(vec)+norm0)
        norm0 = np.linalg.norm(vec)+norm0
    pathxPoints.append(norm0)
    pathx = np.concatenate(pathx)
    numb_values=len(pathx)
    pathk = np.concatenate(pathk)
    
    return pathx, pathk, np.array(pathxPoints)


def sample_square(Lq):
    """Samples the square with edges (-pi,-pi),(-pi,pi),(pi,pi),(pi,-pi) for plotting 2D functions. Note that this is sampling the edge twice.
    Use only for plotting."""
    ks = np.meshgrid(np.linspace(-pi,pi,Lq),np.linspace(-pi,pi,Lq),indexing='ij')
    return np.array(ks)


def sample_reduced_square(Lq):
    """Samples the square with edges (0,pi),(pi,0),(0,-pi),(-pi,0) for plotting 2D functions. Note that this is sampling the edge twice.
    Use only for plotting."""
    ks = np.meshgrid(np.linspace(0,pi,Lq),np.linspace(0,pi,Lq),indexing='ij')
    rotation_matrix = np.array([[1,1],[1,-1]])/2
    ks = np.einsum('ij,jkl->ikl',rotation_matrix,ks)
    return ks


def extent(Lk_or_kmesh):
    """
    Compute the extent for imshow, accounting for pixel boundaries.

    Parameters
    ----------
    Lk_or_kmesh : int or array-like
        Either:
        - int: number of points along each axis (coordinates from -pi to pi)
        - meshgrid array [kx, ky] of shape (2, Nx, Ny)

    Returns
    -------
    list of floats: [xmin, xmax, ymin, ymax]
    """
    
    # Case 1: integer Lk
    if isinstance(Lk_or_kmesh, int):
        Lk = Lk_or_kmesh
        if Lk < 2:
            raise ValueError("Lk must be at least 2.")
        coords = np.linspace(-np.pi, np.pi, Lk)
        dx = coords[1] - coords[0]
        xmin, xmax = coords[0] - dx/2, coords[-1] + dx/2
        ymin, ymax = xmin, xmax
    
    # Case 2: meshgrid input
    elif isinstance(Lk_or_kmesh, (list, tuple, np.ndarray)):
        kmesh = [np.array(k) for k in Lk_or_kmesh[:2]]
        if kmesh[0].ndim != 2 or kmesh[1].ndim != 2:
            raise ValueError("Meshgrid arrays must be 2D.")
        dx = kmesh[0][1,0] - kmesh[0][0,0]
        dy = kmesh[1][0,1] - kmesh[1][0,0]
        xmin, xmax = kmesh[0][0,0]-dx/2, kmesh[0][-1,0]+dx/2
        ymin, ymax = kmesh[1][0,0]-dy/2, kmesh[1][0,-1]+dy/2
    
    # Case 3: anything else → error
    else:
        raise ValueError(f"Invalid input type: {type(Lk_or_kmesh)}. Must be int or meshgrid array.")
    
    return [xmin, xmax, ymin, ymax]


