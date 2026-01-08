import numpy as np
from numpy import pi,cos,sin,exp
import matplotlib
import matplotlib.pyplot as plt

from scipy import interpolate as interpolate #for LL dispersion


def areaOrbitIn(E,Nlin=100,n1=np.array([2*pi,0]),n2=np.array([0,2*pi]),n0=np.array([0,0]),mus=[0],areaBZ=4*pi**2,show=False,**args):
    """Computes the area/BZ of all closed orbits of the reduced Brillouin zone spanned by n1,n2 around n0. Important: Make sure only a single orbit lies in this region and is closed!"""
    #E: dispersion relation; function
    #Nlin: linear spacing of the grid; integer
    #n1,n2: reciprocal vector 1,2; nd.array, shape=(2)
    #n0: center of the reduced brillouin zone; nd.array, shape=(2)
    #mus: fillings; nd.array of floats
    #args: arguments of E except first one
    #Returns: relative area(s); nd.array of len(mus)
    n0 = np.array(n0);n1=np.array(n1);n2=np.array(n2)

    ks = np.array([[i * n1 + j * n2 + n0 for i in np.linspace(-0.5, 0.5, Nlin+1)] for j in np.linspace(-0.5, 0.5, Nlin)])
    [X, Y] = ks.transpose((2, 1, 0))  # the grid for contour
    
    ks = np.swapaxes(ks,0,2)
    
    cs = plt.contour(X.T, Y.T, E(*ks,**args).T, levels=mus)
    plt.clabel(cs, inline=False, fontsize=10)
    areas = []
    for i in range(len(mus)):# Get one of the contours from the plot.
        contour = cs.collections[i]
        if len(contour.get_paths())>0:
            x = np.array([])
            y = np.array([])
            for path in contour.get_paths():
                x = np.concatenate((x,path.vertices[:, 0]))
                y = np.concatenate((y,path.vertices[:, 1]))
            
            x = np.concatenate((x,[x[0]])) #Periodic
            y = np.concatenate((y,[y[0]])) #Periodic
                
            plt.plot(x,y,'k-')
            
            area = 0.5*np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y))
            a = np.abs(area)/areaBZ  # area of orbit/ area of full brillouin zone
            areas.append(a)
        else:
            areas.append(0)
    if show: #show plot and area of the ks
        xy = np.array([n0+n1/2+n2/2,n0+n2/2-n1/2,n0-n1/2-n2/2,n0+n1/2-n2/2,n0+n1/2+n2/2])
        #plt.plot(ks[:,:,0].flatten(),ks[:,:,1].flatten())
        plt.plot(xy[:,0],xy[:,1],'--',color='gray')
        plt.show() # if done correctly now all orbits should appear closed
    #plt.close()
    return np.array(areas)


def return_LL_dispersion(E,mus=np.linspace(-4,0,101),Nlin=100,show=False,**param):
    """Gives the LL dispersion for a FS, i.e. a function taking (l+phase)*B as input and giving the energy of the corresponding LL. 
    The FS is defined by the energy function E, calls areaOrbitIn
    """
    Ss = areaOrbitIn(E,**param,mus=mus,Nlin=Nlin,show=False)
    plt.close()
    e_l = interpolate.interp1d(Ss,mus,fill_value=np.nan,bounds_error=False)
    
    if show:
        plt.plot(Ss,mus)
        plt.xlabel('freq.')
        plt.ylabel('energy')
        plt.show()
    return e_l #e_l = e0_l((l+0.5)*B)


def compute_MB_B(E1,E2,Nlin=100,n1=np.array([2*pi,0]),n2=np.array([0,2*pi]),n0=np.array([0,0]),show=False,**args):
    """Computes the critical flux eB/2*pi of MB junction spanned by n1,n2 around n0. Important: Make sure only two FS lines lay inside!"""
    #E: dispersion relation; function
    #Nlin: linear spacing of the grid; integer
    #n1,n2: reciprocal vector 1,2; nd.array, shape=(2)
    #n0: center of the reduced brillouin zone; nd.array, shape=(2)
    #mus: fillings; nd.array of floats
    #args: arguments of E except first one
    #Returns: relative area(s); nd.array of len(mus)
    n0 = np.array(n0);n1=np.array(n1);n2=np.array(n2)

    ks = np.array([[i * n1 + j * n2 + n0 for i in np.linspace(-0.5, 0.5, Nlin+1)] for j in np.linspace(-0.5, 0.5, Nlin)])
    [X, Y] = ks.transpose((2, 1, 0))  # the grid for contour
    
    ks = np.swapaxes(ks,0,2)
    
    cs1 = plt.contour(X.T, Y.T, E1(*ks,**args).T, levels=[0])
    cs2 = plt.contour(X.T, Y.T, E2(*ks,**args).T, levels=[0])
    #plt.clabel(cs, inline=False, fontsize=10)
    
    points1 = cs1.collections[0].get_paths()[0].vertices #an array [N1,2] containing all points
    points2 = cs2.collections[0].get_paths()[0].vertices #an array [N2,2] containing all points
    
    dks = np.linalg.norm(points1[:,np.newaxis,:] - points2[np.newaxis,:,:],axis=-1)
    [j1,j2] = np.unravel_index(np.argmin(dks), dks.shape)
    k_g = dks[j1,j2] #the minimal distance between the orbits
    
    if show:
        plt.plot(*points1.T,'o')
        plt.plot(*points2.T,'o')
        plt.plot([points1[j1,0]],[points1[j1,1]],'k*')
        plt.plot([points2[j2,0]],[points2[j2,1]],'k*')
        plt.show()
    plt.close()
    
    if len(points1)<=1 or len(points2)<=1:
        return np.nan
    curvatures = []
    for [xs,ys],j in zip([points1.T,points2.T],[j1,j2]):
        Dx = np.mean([xs[j+1]-xs[j],xs[j]-xs[j-1]])
        Dy = np.mean([ys[j+1]-ys[j],ys[j]-ys[j-1]])
        D2x = xs[j+1]+xs[j-1]-2*xs[j]
        D2y = ys[j+1]+ys[j-1]-2*ys[j]
        curvatures.append(np.abs(Dx*D2y-Dy*D2x)/(Dx**2+Dy**2)**(3/2))
    
    B0 = 1/2 * (k_g**3/np.sum(curvatures))**(1/2)
    return B0