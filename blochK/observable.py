import numpy as np
from numpy import pi,cos,sin,exp
from blochK.hamiltonian import Hamiltonian2D


#############################################################################################################################################
#eigenvalues, eigenstates, and expectation values
#############################################################################################################################################


#expectation value of operators
def exp_value_O(O,psi):
    """evalute the expectation value of an Operator O for a set of states psi (band x kys (x kxs) x localH)
    Input:
    O: ndarray, shape=(localH) or (localH x localH)
    """
    if len(O.shape)==1: # O.shape = (localH)
        return exp_value_Odiag(O,psi) #call the faster diagnoal version
    
    # O.shape = (localH1 x localH2)
    if len(psi.shape) == 3:
        res = np.einsum('akb,bc,dkc->adk',np.conjugate(psi),O,psi)
    elif len(psi.shape) == 4:
        res = np.einsum('akqb,bc,dkqc->adkq',np.conjugate(psi),O,psi)
    
    return res #.shape=band1 x band2 x kys (x kys) or band1 x kys (x kys)


def exp_value_Odiag(O,psi):
    """evalute the expectation value of an Operator O (.shape=(n)) for a set of states psi (band x kys (x kxs) x localH)"""
    return np.sum((np.abs(psi)**2*O),axis=-1)


#############################################################################################################################################
#degeneracy of eigenstates
#############################################################################################################################################s

def isDegenerate_1D(es):
    """Checks if an energie in es (.shape=band)"""
    es = np.round(es,10) #to evade numerical errors
    
    single_es, indices, counts = np.unique(es,return_counts=True,return_inverse=True)
    counts = counts>1
    
    condlist = [indices==i for i in range(0,len(counts))]
    xs = np.select(condlist,counts)
    return xs


isDegenerate_vectorized = np.vectorize(isDegenerate_1D,doc='vectorized version of isDegenerate_1D is forwarded to isDegenerate',signature='(n)->(n)') #could not find another way yet


def isDegenerate(es):
    """Checks if an energie in es (.shape=band x ...) is degenerate, i.e. appears more than once in the spectrum"""
    es = np.moveaxis(es,0,-1) #move band to last axis
    res = isDegenerate_vectorized(es) #axis needs to be last otherwise doesn't work
    return np.moveaxis(res,-1,0)


def isDegenerateIn_1D(es, observable_values, threshold=10):
    """Checks if an energie in es (.shape=band)"""
    es = np.round(es,threshold) #to evade numerical errors
    
    single_es, indices, counts = np.unique(es,return_counts=True,return_inverse=True)
    
    selected_sum = np.ones_like(es)
    for i in range(0,len(counts)): #sum over the observable values*index. degenerate states under observable values are degenerate
        selected_sum = np.where(indices==i,np.sum(observable_values*(indices+1),where=indices==i),selected_sum)
        
    return np.abs(selected_sum)<1e-10


isDegenerateIn_vectorized = np.vectorize(isDegenerateIn_1D,doc='vectorized version of isDegenerateIn_1D is forwarded to isDegenerateIn',signature='(n),(n)->(n)',excluded=['threshold']) #could not find another way yet


def isDegenerateIn(es,observable_values,threshold=10):
    """Check if an energie in es (.shape=band x ...) is degenerate in observable_values (.shape=band x ...) which is determined from exp_value_O(xxx_operator)"""
    es = np.moveaxis(es,0,-1) #move band to last axis
    observable_values = np.moveaxis(observable_values,0,-1) #move band to last axis
    res = isDegenerateIn_vectorized(es,observable_values,threshold=threshold) #axis needs to be last otherwise doesn't work
    return np.moveaxis(res,-1,0)




#############################################################################################################################################
#conductivity
#############################################################################################################################################
#conductivity derived from Kubo formula
#both methods give same results,in 'conductivity' the operator is directly evaluated, and 'conductivity_orbital_resolved' return a tensor where one leg can be used to contract with an diagonal operator
#in principle one can write a third function to evalute the conductivity with a non-diagonal operator

def conductivity(Hamiltonian: Hamiltonian2D,Gamma=None,energy=0,operator=None,Lk=50,optimize='path'):
    """
    Evalutes the conductivity with respect to an operator of Hamiltonian_fct with 'Hparam'. 
    Parameters:
    'Hamiltonian_fct': function that returns the Hamiltonian in k-space
    'Hparam': parameters for the Hamiltonian function
    'Gamma':  spectral broadening
    'energy': addtional energy at which the conductivity is evaluated. Default is 0 (Fermi level)
    'operator': the operator with respect to which the conductivity is evaluated. .shape = (localH) or (localH x localH). Default is the identity operator
    'Lq': number of k-points in the q-direction
    'kmesh_BZ': the k-points of the Brillouin zone. If None, square BZ is sampled with 100x100 points
    'optimize': optimization strategy for the computation, see numpy.einsum documentation, for a new problem use 'find_path' first, store it in function and use 'path'
    """
    #sampling the BZ
    ks = Hamiltonian.BZ.sample(Lk)

    #compute the hamiltonian, eigenvalues and eigenstates
    Hk = Hamiltonian.evaluate(*ks) #.shape = (localH,localH,k,q)
    es,psi = Hamiltonian.diagonalize(*ks) #.shape=(band,k,q,localH)

    #define broadening if not given
    if Gamma is None:
        Gamma = find_Gamma(es)

    #compute the derivatives of Hk along unit vectors of BZ
    dk = np.linalg.norm(np.abs(ks[:,0,1]-ks[:,0,0]),axis=0) 
    v1 = -(np.roll(Hk,1,axis=2)-np.roll(Hk,-1,axis=2))/dk/2 #along first axis
    v2 = -(np.roll(Hk,1,axis=3)-np.roll(Hk,-1,axis=3))/dk/2 #along second axis
    v = np.array([v1,v2]) #.shape = (2,localH,localH,k,q)
    
    #calculate the operator density
    if operator is None: #identity operator if none given
        localH = Hk.shape[0]
        operator = np.ones(localH) 
    if len(operator.shape)==1: #operator.shape = (localH)
        jspin = np.einsum('n,inmkq->inmkq',operator,v) #.shape = (2,localH,localH,k,q)
    else: #operator.shape = (localH,localH)
        jspin = np.einsum('ln,inmkq->ilmkq',operator,v)/2 + np.einsum('inmkq,ml->inlkq',v,operator)/2 #.shape = (2,localH,localH,Lq,Lq) #antisymmetrized version if s,
    
    Greenfct = Gamma/((es-energy)**2+Gamma**2) #.shape = (band,Lq,Lq)

    #compute the product of all these quantities
    #contracting of many indices might be costly, therefore use preoptimized path or 'greedy'
    if optimize=='path':
        opt_path = ['einsum_path', (0, 6), (1, 5), (1, 5), (0, 4), (1, 2), (0, 2), (0, 1)]
        sigma = np.einsum('nkqa,iabkq,mkqb,mkqc,jcdkq,nkqd,nkq,mkq->ij',np.conjugate(psi),jspin,psi,np.conjugate(psi),v,psi,Greenfct,Greenfct,optimize=opt_path)
    elif optimize=='find_path': #returns the optimal path, no results!
        opt_path = np.einsum_path('nkqa,iabkq,mkqb,mkqc,jcdkq,nkqd,nkq,mkq->ij',np.conjugate(psi),jspin,psi,np.conjugate(psi),v,psi,Greenfct,Greenfct, optimize='optimal')[0]
        print('Optimal contraction path found:',opt_path)
        return opt_path
    elif optimize=='greedy':
        sigma = np.einsum('nkqa,iabkq,mkqb,mkqc,jcdkq,nkqd,nkq,mkq->ij',np.conjugate(psi),jspin,psi,np.conjugate(psi),v,psi,Greenfct,Greenfct,optimize='greedy')
    else:
        sigma = np.einsum('nkqa,iabkq,mkqb,mkqc,jcdkq,nkqd,nkq,mkq->ij',np.conjugate(psi),jspin,psi,np.conjugate(psi),v,psi,Greenfct,Greenfct)
    
    
    return np.real(sigma)/Lk**2 /np.pi


def conductivity_orbital_resolved(Hamiltonian: Hamiltonian2D,Gamma=None,energy=0,Lk=50,optimize='path'):
    """
    Evalutes the conductivity of Hamiltonian_fct with 'Hparam' in the diagonal bloch basis,
    i.e. the current operator j_iab(k) = O_a * v_ab(k) is not contracted over a (localH index). Only valid for O diagonal. This is much faster than calling conductivity several times.
    Parameters:
    'Hamiltonian_fct': function that returns the Hamiltonian in k-space
    'Hparam': parameters for the Hamiltonian function
    'Gamma':  spectral broadening
    'energy': addtional energy at which the conductivity is evaluated. Default is 0 (Fermi level)
    'Lq': number of k-points in the q-direction
    'kmesh_BZ': the k-points of the Brillouin zone. If None, square BZ is sampled with 100x100 points
    'optimize': optimization strategy for the computation, see numpy.einsum documentation, for a new problem use 'find_path' first, store it in function and use 'path'
    Returns:
    conductivity tensor .shape=(localH,2,2) (basis of H, n1 direction, n2 direction)
    """
    #sampling the BZ
    ks = Hamiltonian.BZ.sample(Lk)

    #compute the hamiltonian, eigenvalues and eigenstates
    Hk = Hamiltonian.evaluate(*ks) #.shape = (localH,localH,k,q)
    es,psi = Hamiltonian.diagonalize(*ks) #.shape=(band,k,q,localH)

    #define broadening if not given
    if Gamma is None:
        Gamma = find_Gamma(es)
    
    #compute the derivatives of Hk along unit vectors of BZ
    dk = np.linalg.norm(np.abs(ks[:,0,1]-ks[:,0,0]),axis=0) 
    v1 = -(np.roll(Hk,1,axis=2)-np.roll(Hk,-1,axis=2))/dk/2 #along first axis
    v2 = -(np.roll(Hk,1,axis=3)-np.roll(Hk,-1,axis=3))/dk/2 #along second axis
    v = np.array([v1,v2]) #.shape = (2,localH,localH,k,q)
        
    Greenfct = Gamma/((es-energy)**2+Gamma**2) #.shape = (band,Lq,Lq)

    #compute the product of all these quantities
    #contracting of many indices might be costly, therefore use preoptimized path or 'greedy'
    if optimize=='path':
        opt_path = ['einsum_path', (0, 6), (1, 5), (1, 5), (0, 4), (1, 2), (0, 2), (0, 1)]
        sigma = np.einsum('nkqa,iabkq,mkqb,mkqc,jcdkq,nkqd,nkq,mkq->aij',np.conjugate(psi),v,psi,np.conjugate(psi),v,psi,Greenfct,Greenfct,optimize=opt_path)
    elif optimize=='find_path': #returns the optimal path, no results!
        opt_path = np.einsum_path('nkqa,iabkq,mkqb,mkqc,jcdkq,nkqd,nkq,mkq->aij',np.conjugate(psi),v,psi,np.conjugate(psi),v,psi,Greenfct,Greenfct, optimize='optimal')[0]
        print('Optimal contraction path found:',opt_path)
        return opt_path
    elif optimize=='greedy':
        sigma = np.einsum('nkqa,iabkq,mkqb,mkqc,jcdkq,nkqd,nkq,mkq->aij',np.conjugate(psi),v,psi,np.conjugate(psi),v,psi,Greenfct,Greenfct,optimize='greedy')
    else:
        sigma = np.einsum('nkqa,iabkq,mkqb,mkqc,jcdkq,nkqd,nkq,mkq->aij',np.conjugate(psi),v,psi,np.conjugate(psi),v,psi,Greenfct,Greenfct)
    
    
    return np.real(sigma)/Lk**2 /np.pi


#############################################################################################################################################
#Computing Quasiparticle Interference (QPI) / Local Density of States (LDOS)
#############################################################################################################################################
#rho(q) = -1/pi Im Tr[ G(k+q) G(k) T], where T = (1 - V0 G_local)^(-1) V0
#
#for V(r) = delta(r) V0, i.e. a local impurity potential, V0 is a matrix of shape (n_orbitals,n_orbitals)
#and G_local = sum_k G(k) appears because of V(r) = delta(r) V0 -> V(q) = V0
#where G(k) = 1/(E-H(k)+i Gamma) is the Green's function
#V0 = operator in the code below

from blochK.utils.cross_correlation import cross_correlation


def local_dos_QPI(Hamiltonian: Hamiltonian2D, Gamma=None,operator=0,Lk=50,kmesh=None,return_symmetric_array=False):
    """
    Computes the local density of states rho(q) in momentum space using T-matrix formalism, i.e.,
    rho(q) = -1/pi * Im Tr[ G(k+q) G(k) T] where T = (1 - V G_local)^(-1) V
    where G(k) = 1/(E-H(k)+i Gamma) is the Green's function, and assumes V(r) = delta(r) V therefore, G_local = sum_k G(k)
    -----------
    Parameters:
    Hamiltonian: Hamiltonian2D object
    Gamma: float
        Broadening of the Green's function. If None it is determined using find_Gamma
    operator: ndarray or float
        operator V0 with shape must be (localH,localH) or a float 
    Lk: int
        Number of k-points in each direction
    kmesh: ndarray
        k-points of the Brillouin zone, shape=(2,Lk,Lk). If None, BZ is sampled
    return_symmetric_array: bool
        If True and Lk even, returns a (Lk+1,Lk+1) array that is symmetric and can be plotted with imshow such that the plot is symmetric
    -----------
    Returns:
    ldos: ndarray
        Local density of states rho(q) in momentum space. Shape is (Lk,Lk) if return_symmetric_array=False, else (Lk+1,Lk+1)
    """
    if kmesh is None:
        kmesh = Hamiltonian.BZ.sample(Lk)
    if isinstance(operator,float) or isinstance(operator,int) :
        operator = operator*np.eye(Hamiltonian.n_orbitals)

    es,psi = Hamiltonian.diagonalize(*kmesh) #psi.shape = band x kys (x kxs) x localH
    dimH = Hamiltonian.n_orbitals #dimension of local Hamiltonian

    assert operator.shape == (dimH,dimH), "operator must be of shape (Hamiltonian2D.n_orbitals,Hamiltonian2D.n_orbitals)"

    #define broadening if not given
    if Gamma is None:
        Gamma = find_Gamma(es)

    #define Green's function tensor, ky momentum, kx momentum, orbital a, orbital b
    opt_path = ['einsum_path', (0, 2), (0, 1)]
    Greensfct_yxab = np.einsum('nyxa,nyxb,nyx->yxab',psi,np.conj(psi),1/(es+1j*Gamma),optimize=opt_path)

    #compute transfer matrix T
    Greensfct_local_ab = np.sum(Greensfct_yxab,axis=(0,1))/Lk**2
    T = np.linalg.inv(np.eye(dimH) - np.dot(operator,Greensfct_local_ab)).dot(operator) #T.shape=(localH,localH)

    #compute z_ac(q) = G_ab(k+q)*G_bc(k) using FFT
    z = cross_correlation(Greensfct_yxab) #z.shape=(qy,qx,a,b)

    #computes local density of states
    ldos = -1/np.pi * np.imag(np.einsum('yxab,ba->yx',z,T)) #+ makes the ldos positive if everything is converged

    ldos = ldos/2 #we computed things for the full BZ, so we double counted, this Hamiltonian_fct dependend
    ldos = np.fft.fftshift(ldos) #shift zero frequency to the center

    #add such that the plot is symmetric
    if return_symmetric_array and Lk%2==0:
        ldos = np.append(ldos, [ldos[0]], axis=0) #append first row at the end
        ldos = np.append(ldos, ldos[:,:1], axis=1) #append first column at the end

    #things to check:
    #for converged results, ldos should be positive everywhere
    #for FS Gamma centered, FS should match with peak of imshow with extent=(-pi,pi,-pi,pi)/2 

    return ldos

#############################################################################################################################################
#Computing Magnetic Linear Dichroism
#############################################################################################################################################
# MLD(w) = I_x(w) - I_y(w)
# I_a(w) = sum_{n,m,k} |M_a|^2 delta(E_m-E_n - w) n_FD(E_n) (1-n_FD(E_m))
# where M_a = |<m|v_a|n>| and v_a = dH/dk_a

def magnetic_linear_dichroism(Hamiltonian:Hamiltonian2D, omegas:np.ndarray, Lk=100, fact_eps=1):
    """
    Compute the magnetic linear dichroism spectrum of a 2D Hamiltonian.
    Parameters:
    Hamiltonian: Hamiltonian2D
        The Hamiltonian for which to compute the spectrum.
    omegas: np.ndarray
        Array of frequency values at which to compute the spectrum.
    Lk: int
        Number of k-points along each axis in the Brillouin zone sampling.
    fact_eps: float
        Determines broadening of delta function, see below.
    Returns:
    intensity: np.ndarray.shape=(2,len(omegas))
        Magnetic linear dichroism intensity in m1 and m2 direction, as a function of frequency.
    """

    #sampling the BZ
    ks = Hamiltonian.BZ.sample(Lk)

    #compute the hamiltonian, eigenvalues and eigenstates
    Hk = Hamiltonian.evaluate(*ks) #.shape = (localH,localH,k,q)
    es,psi = Hamiltonian.diagonalize(*ks) #.shape=(band,k,q,localH)

    #compute the derivatives of Hk along unit vectors of BZ
    dk = np.linalg.norm(np.abs(ks[:,0,1]-ks[:,0,0]),axis=0) 
    v1 = -(np.roll(Hk,1,axis=2)-np.roll(Hk,-1,axis=2))/dk/2 #along first axis
    v2 = -(np.roll(Hk,1,axis=3)-np.roll(Hk,-1,axis=3))/dk/2 #along second axis
    v = np.array([v1,v2]) #.shape = (2,localH,localH,k,q)

    #compute the matrix elements between bands
    M_amnkq = np.einsum('mkqi,aijkq,nkqj->amnkq',np.conjugate(psi),v,psi) #.shape = (2,band,band,k,q)

    #occupied states
    fermi_occupation = es<0 #.shape = (band,k,q)

    #delta function
    #delta(E_m-E_n-omega) 
    #using Lorentzian representation
    #delta(x-w) = eps/(x^2 + eps^2)/pi
    #eps = omega_spacing * fact_eps
    eps = fact_eps*(omegas[1]-omegas[0]) #broadening for delta function
    de_mnkqw = np.expand_dims(np.expand_dims(es,1)-np.expand_dims(es,0),-1) - omegas #de = E_m-E_n-omega; de.shape = (m,n,k,q,omegas)
    delta_mnkqw = eps/(de_mnkqw**2 + eps**2)/np.pi #.shape = (m,n,k,q,omegas)

    #sum: bands n,m, momentum k,q
    intensity_aw = np.einsum('amnkq,mnkqw,nkq,mkq->aw',np.abs(M_amnkq)**2,delta_mnkqw,fermi_occupation,1-fermi_occupation)/Lk**2

    #difference of x and y absorption
    MLD_intensity = intensity_aw[0]-intensity_aw[1]

    return MLD_intensity


#############################################################################################################################################
#helper functions
#############################################################################################################################################

def find_Gamma(es, weighting_factor=0.8):
    """Defines a suitable broadening Gamma based on the maximum variation of the band structure es."""
    des_x = np.roll(es, 1, axis=2) - es
    des_y = np.roll(es, 1, axis=1) - es
    des = np.abs([des_x, des_y]).flatten()
    des0 = des.max()
    return des0 * weighting_factor
