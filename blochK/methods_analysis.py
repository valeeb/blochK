import numpy as np
import matplotlib.pylab as plt

def find_phase_boundary(order_parameter,xs,ys,eps=1e-3):
    """Determines the points at which the order_parameter (.shape=(X,Y)) becomes non-zero. Coordinates of this array are xs (.shape=(X)), ys (.shape=(Y))"""

    coords = np.swapaxes(np.meshgrid(xs,ys,indexing='ij'),0,-1)

    order_parameter = 1*(np.abs(order_parameter)>eps) #True if inside the phase

    #check where phase boundary is
    jump0 = np.array(np.abs(order_parameter[1:]-order_parameter[:-1]),dtype=bool) #take the derivative
    coordinates0 = (coords[1:]+coords[:-1])/2 #compute the x,y coordinates of the derivative
    jump1 = np.array(np.abs(order_parameter[:,1:]-order_parameter[:,:-1]),dtype=bool)
    coordinates1 = (coords[:,1:]+coords[:,:-1])/2
    #list of x,y values phase boundary
    phase_boundary_xy = np.concatenate((coordinates0[jump0],coordinates1[jump1])) #.shape=(2,.)
    asort = np.argsort(1j*phase_boundary_xy[:,0]+phase_boundary_xy[:,1]) #sort the list by its y-values first, then by its x values.
    phase_boundary = np.array([phase_boundary_xy[:,0][asort],phase_boundary_xy[:,1][asort]]) #.shape=(.,2) sorted by x-value
    
    return phase_boundary


def find_phase_boundary2(order_parameter,x=None,y=None,threshold_value=1e-3,show=False):
    """Find the phase boundary based on a contour plot"""
    fig,ax = plt.subplots(1,1)
    
    cs = ax.contour(x,y, order_parameter, levels=[threshold_value])
    contour = cs.collections[0]
    coord = contour.get_paths()[0].vertices[:, :].T
    
    if show:
        fig.show()
    else:
        plt.close()
    
    return coord #.shape=(2,.)