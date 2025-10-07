import numpy as np
from numpy import pi,cos,sin,exp
import matplotlib
import copy
import matplotlib.pyplot as plt

from blochK.plotting.utils import path,sample_square
from blochK.observable import exp_value_O, isDegenerateIn

#for coloring lines
from matplotlib.collections import LineCollection


def plot_FS(ax,Hamiltonian, Lk=200, coloring_operator='k',show_xlabel=True,show_ylabel=True,show_FS=True,cmap='none',print_filling=False,kmesh='square'):
    """
    Plots Fermi surface of Hamiltonian on ax
    Parameters:
    ax: matplotlib axis
    Hamiltonian: Hamiltonian2D object
    Lk: number of k points along each direction. Mutually exclusive with np.ndarray form of kmesh
    kmesh: 'square' (default) or 'BZ' or np.ndarray of shape (2,Lkx,Lky) with kx,ky points
    coloring operator: a color (fixed color of all bands) or an operator (colored by eigenvalues), i..e. ndarraty of shape (Hamiltonian.n_orbitals,Hamiltonian.n_orbitals) or (Hamiltonian.n_orbitals,)
    """
    #check coloring operator
    if isinstance(coloring_operator,str):
        assert isinstance(coloring_operator,str), 'coloring operator must be a color (string) or an operator (ndarray) with shape matching the Hamiltonian'
    elif isinstance(coloring_operator,np.ndarray):
        assert coloring_operator.shape == (Hamiltonian.n_orbitals,Hamiltonian.n_orbitals) or coloring_operator.shape == (Hamiltonian.n_orbitals,), 'coloring operator must be an operator (ndarray) with shape matching the Hamiltonian'
    else:
        raise ValueError('coloring operator must be a color (string) or an operator (ndarray) with shape matching the Hamiltonian')
    
    #setting a nice colormap
    if cmap=='none':
        cmap = copy.copy(matplotlib.colormaps["brg"]) #set the name of the colormap
    else:
        cmap = copy.copy(matplotlib.colormaps[cmap]) #set the name of the colormap
    cmap.set_under(color='black')
    cmap.set_over(color='gray')
    norm = plt.Normalize(0, 1) # Create a continuous norm to map from data points to colors

    if kmesh=='BZ':
        bBZ = Hamiltonian.BZ.return_boundary()
        max = np.abs(bBZ).max()
        ks = np.meshgrid(np.linspace(-max,max,Lk),np.linspace(-max,max,Lk),indexing='ij')
        ks = np.array(ks)
    elif kmesh=='square':
        ks = sample_square(Lk)
        max = pi
    else:
        assert isinstance(kmesh,np.ndarray) and kmesh.shape[0]==2, 'kmesh must be "square", "BZ" or an ndarray of shape (2,Lkx,Lky)'
        ks = kmesh
    xs = ks[0]; ys = ks[1]

    es,_ = Hamiltonian.diagonalize(*ks)
    
    ax.set_aspect('equal')
    if show_xlabel:
        ax.set_xlabel(r'$k_x$',labelpad=1)
        ax.set_xticks([-pi,0,pi])
        ax.set_xticklabels([r'$-\pi$',0,r'$\pi$'])
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel(r'$k_y$',labelpad=-2.5)
        ax.set_yticks([-pi,0,pi])
        ax.set_yticklabels([r'$-\pi$',0,r'$\pi$'])
    else:
        ax.set_yticklabels([])
    
    ax.set_xlim(-max,max)
    ax.set_ylim(-max,max)
    ax.set_xticks([-pi,0,pi])
    ax.set_yticks([-pi,0,pi])
    #--------------
    if show_FS:
        for iband in range(len(es)): #for each band
            if isinstance(coloring_operator,str): #if coloring operator is a color
                ax.contour(xs,ys,es[iband],[0],colors=coloring_operator)
            else: #if coloring operator is an operator
                FS0 = ax.contour(xs,ys,es[iband],[0],alpha=0)
                for datapoints in FS0.allsegs[0]:
                    if datapoints.shape[0]>2:
                        points = datapoints.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1) #create a list of N-1 lines form v0 to v1, from v1 to v2,...
                        es_FS,psis_FS = Hamiltonian.diagonalize(datapoints[:,0],datapoints[:,1]) #determine es, psis along contour
                        Os = exp_value_O(coloring_operator,psis_FS)
                        isDeg = isDegenerateIn(es_FS,Os,threshold=3)
                        lc = LineCollection(segments,cmap=cmap,norm=norm, capstyle='projecting')

                        normalized_color = (1-1e-10)*(Os[iband]+1)/2 + 1e-15 - 10*isDeg[iband] # shift regular values from interval [0,1] to (0,1) by infinitesimals, make degnerate entries negative

                        lc.set_array(normalized_color) #Set the values used for colormapping
                        line = ax.add_collection(lc)
    #----------------
    if print_filling:
        print('filling is: ', np.sum(es<0)/np.prod(es.shape)) #filling

                    

def plot_bandstruc(ax,Hamiltonian,points_path=None, labels_points_path=[r'\Gamma','X','R','Y',r'\Gamma'],N_samples=100, coloring_operator='k',show_xlabel=True,show_ylabel=True,cmap='none'):
    """
    Plots Fermi surface of Hamiltonian on ax
    Parameters:
    ax: matplotlib axis
    Hamiltonian: Hamiltonian2D object
    N_samples: number of k points between each point in points_path
    points_path: list of k-points defining the path in the BZ, default None, in which case it is taken from labels_points_path
    labels_points_path: list of labels for the k-points defining the path in the BZ
    coloring operator: a color (fixed color of all bands) or an operator (colored by eigenvalues), i..e. ndarray of shape (Hamiltonian.n_orbitals,Hamiltonian.n_orbitals) or (Hamiltonian.n_orbitals,)
    """

    #if no path is given, get it from the labels, assuming they are in the BZ points
    if points_path is None: 
        if not all([p in Hamiltonian.BZ.points.keys() for p in labels_points_path]):
            raise ValueError('all labels in labels_points_path must be in Hamiltonian.BZ.points')
        points_path = [Hamiltonian.BZ.points[p] for p in labels_points_path]
        labels_points_path=[r'${}$'.format(l) for l in labels_points_path] #transform to latex format

    #setting the path
    ts, ks, ticks = path(points_path,N_samples=N_samples)

    #setting a nice colormap
    if cmap=='none':
        cmap = copy.copy(matplotlib.colormaps["brg"]) #set the name of the colormap
    else:
        cmap = copy.copy(matplotlib.colormaps[cmap]) #set the name of the colormap
    cmap.set_under(color='black')
    cmap.set_over(color='gray')
    norm = plt.Normalize(0, 1) # Create a continuous norm to map from data points to colors
    
    es,psis = Hamiltonian.diagonalize(ks[:,0],ks[:,1])

    ax.axhline(0,linestyle='--',color='k',zorder=0)

    if isinstance(coloring_operator,str): #if coloring operator is a color
        for iband in range(len(es)):
            ax.plot(ts,es[iband],color=coloring_operator,zorder=0)
    else: #if coloring operator is an operator
        #plot band with coloring given by Ss
        Os = exp_value_O(np.array(coloring_operator),psis)
        isDeg = isDegenerateIn(es,Os,threshold=3)

        for iband in range(len(es)):
            ax.plot(ts,es[iband],alpha=0)
            points = np.array([ts, es[iband]]).T.reshape(-1, 1, 2) #data has N points
            segments = np.concatenate([points[:-1], points[1:]], axis=1) #create a list of N-1 lines form v0 to v1, from v1 to v2,...
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            normalized_color = (1-1e-10)*(Os[iband]+1)/2 + 1e-15 - 10*isDeg[iband] # shift regular values from interval [0,1] to (0,1) by infinitesimals, make degnerate entries negative
            lc.set_array(normalized_color) # Set the values used for colormapping
            line = ax.add_collection(lc)

    #ax.set_ylim(-1,1)
    ax.set_xlim(ts[0],ts[-1])
    ax.set_xticks(ticks)
    
    if show_ylabel:
        ax.set_ylabel(r'$E/t$',labelpad=-4.5)
    else:
        ax.set_yticklabels([])
    if show_xlabel:
        ax.set_xticklabels(labels_points_path)
    else:
        ax.set_xticklabels([])


    