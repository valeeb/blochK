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
    """Samples the square with edges (-pi,-pi),(-pi,pi),(pi,pi),(pi,-pi) for plotting 2D functions. Not that this is sampling the edge twice.
    Use only for plotting."""
    ks = np.meshgrid(np.linspace(-pi,pi,Lq),np.linspace(-pi,pi,Lq),indexing='ij')
    return np.array(ks)


def sample_reduced_square(Lq):
    """Samples the square with edges (0,pi),(pi,0),(0,-pi),(-pi,0) for plotting 2D functions. Not that this is sampling the edge twice.
    Use only for plotting."""
    ks = np.meshgrid(np.linspace(0,pi,Lq),np.linspace(0,pi,Lq),indexing='ij')
    rotation_matrix = np.array([[1,1],[1,-1]])/2
    ks = np.einsum('ij,jkl->ikl',rotation_matrix,ks)
    return ks