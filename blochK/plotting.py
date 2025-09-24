import numpy as np
from numpy import pi,cos,sin,exp
import matplotlib
import copy
import matplotlib.pyplot as plt
from matplotlib import cm

from blochK.methods_basic import path
from blochK.observable import exp_value_O, isDegenerateIn, eigs_H

#for coloring lines
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


def plot_FS(ax,Hamiltonian_fct,param={}, Lq=200, coloring_operator='k',show_title=True,show_xlabel=True,show_ylabel=True,show_FS=True,cmap='none',print_filling=False):
    """Plots colored FS of psik8(param) on ax,

    coloring operator: a color (fixed color of all bands) or an operator (colored by eigenvalues)

    """
    #setting a nice colormap
    if cmap=='none':
        cmap = copy.copy(matplotlib.cm.get_cmap("brg")) #set the name of the colormap
    else:
        cmap = copy.copy(matplotlib.cm.get_cmap(cmap)) #set the name of the colormap
    cmap.set_under(color='black')
    cmap.set_over(color='gray')
    norm = plt.Normalize(0, 1) # Create a continuous norm to map from data points to colors

    kxs = np.linspace(-pi,pi,Lq)
    kys = np.linspace(-pi,pi,Lq)
    xs,ys = np.meshgrid(kxs,kys)
    ks = np.moveaxis([xs,ys],0,-1)

    es,psis = eigs_H(xs,ys,Hamiltonian_fct,param)

    
#     if show_title:
#         ax.set_title('V={:4.2f},  J={:4.2f},\ndn={:4.2f}, dm={:4.2f}'.format(param['V'],param['J'],param['dn'],param['dm']),fontsize=6)
    ax.set_aspect('equal')
    if show_xlabel:
        ax.set_xlabel('$k_x$',labelpad=1)
        ax.set_xticks([-pi,0,pi])
        ax.set_xticklabels(['$-\pi$',0,'$\pi$'])
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel('$k_y$',labelpad=-2.5)
        ax.set_yticks([-pi,0,pi])
        ax.set_yticklabels(['$-\pi$',0,'$\pi$'])
    else:
        ax.set_yticklabels([])
    
    ax.set_xlim(-pi,pi)
    ax.set_ylim(-pi,pi)
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
                        es_FS,psis_FS = eigs_H(datapoints[:,0],datapoints[:,1],Hamiltonian_fct,param) #determine es, psis along contour
                        Os = exp_value_O(coloring_operator,psis_FS)
                        isDeg = isDegenerateIn(es_FS,Os,threshold=3)
                        lc = LineCollection(segments,cmap=cmap,norm=norm, capstyle='projecting')

                        normalized_color = (1-1e-10)*(Os[iband]+1)/2 + 1e-15 - 10*isDeg[iband] # shift regular values from interval [0,1] to (0,1) by infinitesimals, make degnerate entries negative

                        lc.set_array(normalized_color) #Set the values used for colormapping
                        line = ax.add_collection(lc)
    #----------------
    if print_filling:
        print(np.sum(es<0)/np.prod(es.shape)) #filling

                    

def plot_bandstruc(ax,Hamiltonian_fct,param={},points_path=[[0,0],[pi,0],[0,pi],[0,0]], labels_points_path=['$[0,0]$','$[pi,0]$','$[0,pi]$','$[0,0]$'],N_samples=100, coloring_operator='k',show_title=True,show_xlabel=True,show_ylabel=True,cmap='none'):
    """Plots colored bandstructure along points_path of psik8(param) on ax,
    
    coloring operator: a color (fixed color of all bands) or an operator (colored by eigenvalues)
    """
    #setting the path
    ts, ks, ticks = path(points_path,N_samples=N_samples)
    
    #setting a nice colormap
    if cmap=='none':
        cmap = copy.copy(matplotlib.cm.get_cmap("brg")) #set the name of the colormap
    else:
        cmap = copy.copy(matplotlib.cm.get_cmap(cmap)) #set the name of the colormap
    cmap.set_under(color='black')
    cmap.set_over(color='gray')
    norm = plt.Normalize(0, 1) # Create a continuous norm to map from data points to colors
    
    es,psis = eigs_H(ks[:,0],ks[:,1],Hamiltonian_fct,param)
    
    
    #print(param['dm'],param['dn'])
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
    
#     if show_title:
#         ax.set_title('dn={0}, dm={1}, (V,J) = ({2},{3})'.format(np.round(param['dn'],4),np.round(param['dm'],4),np.round(param['V'],2),np.round(param['J'],2)))
    if show_ylabel:
        ax.set_ylabel('$E/t$',labelpad=-4.5)
    else:
        ax.set_yticklabels([])
    if show_xlabel:
        ax.set_xticklabels(labels_points_path)
    else:
        ax.set_xticklabels([])

        
        

    