import numpy as np
from numpy import pi,cos,sin,exp
import matplotlib
import copy
import matplotlib.pyplot as plt
import scipy.integrate as integrate

from blochK import Hamiltonian2D


def line_integration(ks,O,closed=True):
    """Computes a line integral along a closed path given by ks (shape=(2,ks)). O (shape=(:,...,:,ks)) is the integrand"""

    # if closed: #we must add the first point to the end to close the path
    #     ks = np.concatenate([ks, ks[:, [0]]], axis=1)
    #     O = np.concatenate([O, O[..., [0]]], axis=-1)

    def xy2path(x,y):
        """Takes x,y (shape=(N)) coordinates of a contour creates a 1D path t (shape=(N), i.e. the length of the contour."""
        dt = ((np.roll(x,1)-x)**2+(np.roll(y,1)-y)**2)**0.5 #compute distance between consecutive points
        t = dt.cumsum() #the distance of each point along the path
        t = t-t[0]
        return t
    
    t = xy2path(*ks)
    I = integrate.simpson(O,x=t,axis=-1)
    return I



def get_points_FS(Hamiltonian:Hamiltonian2D,energys=[0],show=False,Lk:int=100):
    """
    Gives the coordinates of the Fermi surface of a Hamiltonian object 
    Parameters:
    Hamiltonian: Hamiltonian2D object
    energys: list of energies at which to compute the Fermi surface
    show: bool, if True shows the contour plot used to compute the Fermi surface
    Lk: int, number of k points along each direction
    Returns:
    paths:  
        list of lists (not a tensor because different segments can have different lengths)
        axis order is (bands, energys, segments in the BZ, points of each segment, x and y cooridinates)
    """

    ks = Hamiltonian.BZ.sample(Lk=Lk)
    es,_ = Hamiltonian.diagonalize(*ks)

    fig, axs = plt.subplots(1,len(es),figsize=(12,1.5))
    if len(es)==1: axs = [axs]
    
    paths = [] #list of bands, list of energys, list of segments, points, x and y components(2)
    for band in range(es.shape[0]):
        es_band = es[band]
        cs = axs[band].contour(*ks, es_band, levels=energys)
        axs[band].clabel(cs, inline=False, fontsize=10)
        axs[band].set_aspect('equal')
        coords = cs.allsegs #coordinates: energys, number of paths, points, x and y components(2)

        paths.append([])
        for i in range(len(energys)):# Get one of the contours from the plot.
            if len(coords[i])>0:
                paths[-1].append([x.T for x in coords[i]])
            else:
                paths[-1].append(np.array([[[np.nan],[np.nan]]]))
    if show:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


    return paths #list of bands, list of mus, list of segments, x and y components(2), points



def equalizeFS(k_FSs_bands,tol=1e-3):
    """Given a list (bands) of a list(zero energy contours) of k values. Check if some of them are equal within tolerance if yes equalize them"""
    #numpy array including all shapes
    shapes0 = [ks.shape for group in k_FSs_bands for ks in group]
    shapes = np.empty(len(shapes0), dtype=object)
    shapes[:] = shapes0
    #numpy array including the 
    kss0 = [ks for group in k_FSs_bands for ks in group] #basically an array of pointers to the stored k values
    kss = np.empty(len(kss0), dtype=object)
    kss[:] = kss0

    values,inverses = np.unique(shapes,return_inverse=True) #find elements with equal shape

    for index_equal in range(len(values)):
        equal_kss = kss[inverses==index_equal] #elements with equal shape
        for i in range(len(equal_kss)): #check if elements with equal shape are close
            for j in range(i+1,len(equal_kss)):
                if np.allclose(equal_kss[i], equal_kss[j], rtol=1e-08, atol=tol, equal_nan=True):
                    equal_kss[j] = equal_kss[i] #equal elements which are close
                    #print(equal_kss[i].shape,i,j,'equal')
    return k_FSs_bands