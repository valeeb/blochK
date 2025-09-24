import numpy as np
from numpy import pi,cos,sin,exp


#############################################################################################################################################
#eigenvalues, eigenstates, and expectation values
#############################################################################################################################################


#solve a hamiltonian
def eigs_H(kx,ky,Hamiltonian_fct,Hparam): 
    """
    Return energies 'es' and wavefunctions 'psis' in the shape:
    es.shape = band x kys (x kxs)
    psi.shape = band x kys (x kxs) x localH
    """
    Hk = Hamiltonian_fct(kx,ky,**Hparam)
    
    Hk = np.moveaxis(np.moveaxis(Hk,1,-1),0,-2) #make it ...x band x band dimensional
    es,vs = np.linalg.eigh(Hk)
    es = np.moveaxis(es,-1,0) #.shape=band x kys (x kxs)
    psi = np.moveaxis(vs,-1,0) #.shape=band x kys (x kxs) x localH
    
    return es, psi


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

from blochK.methods_basic import sample_BZ

def conductivity(Hamiltonian_fct,Hparam=dict(),Gamma=1e-2,energy=0,operator=None,kmesh_BZ=None,basis='xy',optimize='path'):
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
    if kmesh_BZ is None:
        Lq = 100
        kmesh_BZ = sample_BZ(Lq)
    else:
        Lq = kmesh_BZ.shape[1]
    ks = kmesh_BZ #ks.shape=(2,k,q)

    #compute the hamiltonian, eigenvalues and eigenstates
    Hk = Hamiltonian_fct(*ks,**Hparam) #.shape = (localH,localH,k,q)
    es,psi = eigs_H(*ks,Hamiltonian_fct,Hparam) #.shape=(band,k,q,localH)
    
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
    
    
    return np.real(sigma)/Lq**2 /np.pi


def conductivity_orbital_resolved(Hamiltonian_fct,Hparam=dict(),Gamma=1e-2,energy=0,kmesh_BZ=None,optimize='path'):
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
    if kmesh_BZ is None:
        Lq = 100
        kmesh_BZ = sample_BZ(Lq)
    else:
        Lq = kmesh_BZ.shape[1]
    ks = kmesh_BZ #ks.shape=(2,k,q)

    #compute the hamiltonian, eigenvalues and eigenstates
    Hk = Hamiltonian_fct(*ks,**Hparam) #.shape = (localH,localH,k,q)
    es,psi = eigs_H(*ks,Hamiltonian_fct,Hparam) #.shape=(band,k,q,localH)
    
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
    
    
    return np.real(sigma)/Lq**2 /np.pi
