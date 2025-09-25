import numpy as np
from numpy import pi,cos,sin,exp
import matplotlib
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import scipy.integrate as integrate


def line_integration(ks,O):
    """Computes a line integral along a path given by ks (shape=(2,ks)). O (shape=(:,...,:,ks)) is the integrand"""
    def xy2path(x,y):
        """Takes x,y (shape=(N)) coordinates of a contour creates a 1D path t (shape=(N), i.e. the length of the contour."""
        dt = ((np.roll(x,1)-x)**2+(np.roll(y,1)-y)**2)**0.5 #compute distance between consecutive points
        t = dt.cumsum() #the distance of each point along the path
        t = t-t[0]
        return t
    
    t = xy2path(*ks)
    I = integrate.simps(O,x=t,axis=-1)
    return I


def getPointsFS(E,Nlin=100,n1=np.array([2*pi,0]),n2=np.array([0,2*pi]),n0=np.array([0,0]),show=False,Eargs={}):
    """Get points on the FS inside the region spanned by n1,n2 around n0."""
    #E: dispersion relation; function
    #Nlin: linear spacing of the grid; integer
    #n1,n2: reciprocal vector 1,2; nd.array, shape=(2)
    #n0: center of plotting region
    #mus: fillings; nd.array of floats
    #args: arguments of E except first one
    #Returns: x,y coordinates of the points
    n0 = np.array(n0);n1=np.array(n1);n2=np.array(n2)

    ks = np.array([[i * n1 + j * n2 + n0 for i in np.linspace(-0.5, 0.5, Nlin+1)] for j in np.linspace(-0.5, 0.5, Nlin)])
    [X, Y] = ks.transpose((2, 1, 0))  # the grid for contour
    es = E(ks[:,:,0],ks[:,:,1],Eargs)
    
    coords = []
    fig, axs = plt.subplots(1,len(es),figsize=(12,1.5))
    
    for es_band,ax in zip(es,axs):
        cs = ax.contour(X.T, Y.T, es_band, levels=[0])

        contour = cs.collections[0]
        coord_band = []
        if len(contour.get_paths())>0:
            ########iterate through all paths
            for path in contour.get_paths():
                x = path.vertices[:, 0]
                y = path.vertices[:, 1]
                coord_band.append(np.array([x,y]))    
        else:
            #print('No FS')
            return [np.array([[np.nan],[np.nan]])]

        #show plot and area of the ks
        if show: 
            xy = np.array([n0+n1/2+n2/2,n0+n2/2-n1/2,n0-n1/2-n2/2,n0+n1/2-n2/2,n0+n1/2+n2/2])
            #plt.plot(ks[:,:,0].flatten(),ks[:,:,1].flatten())
            ax.plot(xy[:,0],xy[:,1],'--',color='gray')

        #thin out point density (some points will lay very close to each other, kick them out, they are not needed)
        def select_xy(x,y,eps=1e-8):
            """Takes x,y (shape=(N)) coordinates of a contour and checks if their at least eps apart."""
            dt = ((np.roll(x,1)-x)**2+(np.roll(y,1)-y)**2)**0.5 #compute distance between consecutive points
            select = dt>eps #take only points which do not lay on top of each other
            return select

        for [x,y] in coord_band:
            select = select_xy(x,y)
             #the last point should be the same as the first (PBC)
            x = np.insert(x[select],0,x[0])
            y = np.insert(y[select],0,y[0])
        
        coords.append(coord_band.copy())
        
    if show:
        plt.tight_layout()
        plt.show()
    plt.close(fig)
    
    return coords #not a numpy array ".shape" = (8 bands, list of ks, 2 kx&ky, ks)


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