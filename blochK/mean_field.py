import numpy as np
import scipy.optimize as optimize
from hamiltonian_2orbital import psik8,E8_const,Spin_operator,Sublattice_operator,Orbital_operator
from methods_basic import sample_reducedBZ

def nF(e,T):
    """the Fermi-Dirac distribution function"""
    if T>0.:
        return 1/(1+np.exp(e/T))
    else:
        return np.where(e<0,1,0)


def MFstep(T,mf,Lq,J=0,V=0,mu0=1,mu_tol=1e-4,**param):
    """Performs a single mean-field step, i.e. for given parameters the MF ns of the GS"""
    #ks=np.meshgrid(np.linspace(-pi,pi,Lq),np.linspace(-pi,pi,Lq),indexing='ij') #discretize the Brillouin zone (costs nearly no time)
    ks = sample_reducedBZ(Lq)
    N = int(param['n']*Lq**2) #total number of particles
    
    Hparam = dict(J=J,V=V,dm=mf[0],dn=mf[1],**param)
    es, psis = psik8(*ks,Hparam)
    
    #find mu such that n is fullfilled
    #rootres = optimize.root_scalar(lambda x: np.sum(nF(es-x,T))-N,x0=mu0-1,x1=mu0+1,method='secant',xtol=1e-3) #optimize mu0 by secant method
    rootres = optimize.root_scalar(lambda x: np.sum(nF(es-x,T))-N,bracket=(mu0-30,mu0+30),method='bisect',xtol=mu_tol) #optimize mu0 by bisect method
    mu = rootres.root
    #calculate the energy
    e0 = np.sum(es*nF(es-mu,T))/Lq**2 + E8_const(dm=mf[0],dn=mf[1],V=V,J=J)
    #calculate the densities
    ns = np.sum(np.abs(psis**2)*nF(es-mu,T)[:,:,:,np.newaxis],axis=(0,1,2))/Lq**2
    mf = np.array([Spin_operator*Sublattice_operator,Orbital_operator*Sublattice_operator]).dot(ns)/8
    
    return mf, mu, e0

def convergeMF(T,mf0,Lq,tol=1e-2,max_number_of_steps=200,mu0=0,J=0,V=0,**param):
    """Runs the MF step until MF dn with starting value is converged."""
    mfs=[mf0]
    mus=[mu0]
    es = []
    for i in range(max_number_of_steps):
        mf,mu,e = MFstep(T,mfs[i],Lq,mu0=mus[i],J=J,V=V,**param,mu_tol=tol)
        mfs.append(mf)
        es.append(e)
        mus.append(mu)
        if i>10 and  np.all(np.abs(mfs[-4:]-np.mean(mfs[-4:],axis=0))<tol):#np.abs(dns[i]-dns[i-1])<tol:
            break
    #print('No convergence after 500 steps. Fluctuations around the mean: {3}  \t (U={0}, T={1}, dn0={2})'.format(U,T,dn0,dns[-4:]-np.mean(dns[-4:])))
    
    #add trivial solutions where at least one of the MF is zero
#     for mf0 in np.array([[mfs[-1][0],0],[0,mfs[-1][1]],[0,0]]): #
#         mf,mu,e = MFstep(T,mf0,Lq,mu0=mus[i],J=J,V=V,**param)
#         if np.all(np.abs(mf-mf0)<tol):
#             es.append(e)
#             mus.append(mu)
#             mfs.append(mf)
    
    return np.array(mfs),np.array(mus),i,np.array(es)

def findMF(T,mf0,Lq,tol=1e-2,max_number_of_steps=200,mu0=0,J=0,V=0,**param):
    """Applies the converge MF method and returns the correct MFs"""
    mfs,mus,numb_MFiterations,es = convergeMF(T,mf0,Lq,tol=tol,max_number_of_steps=max_number_of_steps,mu0=mu0,J=J,V=V,**param)
    minimalIndex = es[-5:].argmin()-5 # selects the one with minimal energy
    #minimalIndex = -1
    #a result object with all relevant data
    res ={'mean_fields':mfs.copy(),'mus':mus.copy(),'tolerance':tol,'max_number_of_steps':max_number_of_steps,'Lq':Lq,'T':T,'energies':es.copy(),'MF_iterations':numb_MFiterations}
    res['param'] = dict(**param,J=J,V=V,mu=mus[minimalIndex],dm=mfs[minimalIndex,0],dn=mfs[minimalIndex,1])
    return mfs[minimalIndex],mus[minimalIndex],res