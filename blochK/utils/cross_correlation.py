#This file is exclusively dedicated to compute
#z_ac(q1,q2) = sum_{k1,k2,b} G_{ab}(k1+q1,k2+q2) G_{bc}(k1,k2)
#it includes 4 methods all giving the same result. 2),3), and 4) are for testing purposes
#1) Using the convolution theorem and FFT (very fast)
#2) Using broadcasting (slow)
#3) Using 2*loops + 2*np.roll (very slow)
#4) Using 4*loops (very,very slow)

import numpy as np


def cross_correlation(G: np.ndarray) -> np.ndarray:
    """Computes z_ab(q1,q2) = sum_{k1,k2,c} G_{ac}(k1+q1,k2+q2) G_{cb}(k1,k2)
    using the convolution theorem and FFT
    Parameters:
    G : np.ndarray.shape = (Lq1,Lq2,a,c) which are ky momentum, kx momentum, orbital_out, orbital_in
    Returns:
    z : np.ndarray.shape = (Lq1,Lq2,a,b) which are qy momentum, qx momentum, orbital_out, orbital_in
    """
    (Lq1,Lq2,_,_) = G.shape

    X = np.fft.fft2(G,axes=(0,1)) #X(k) = G(k)
    Y = np.fft.fft2(np.flip(G,axis=(0,1)),axes=(0,1)) #Y(k) = G(-k)
    Z = np.einsum('yxac,yxcb->yxab', X, Y, optimize=['einsum_path', (0, 1)]) #this is the opitmal path I found
    #opt_path = np.einsum_path('yxac,yxcb->yxab', X, Y, optimize='optimal')[0]
    #print('Optimal contraction path found:',opt_path)
    z = np.fft.ifft2(Z,axes=(0,1))/Lq1/Lq2 #z.shape=(qy,qx,a,b)
    z = np.roll(z,1,axis=(0,1)) #I have no idea why but I need to roll by one to get the same result

    return z


def cross_correlation_broadcast(G: np.ndarray) -> np.ndarray:
    """Only for testing.
    Computes z_ab(q1,q2) = sum_{k1,k2,c} G_{ac}(k1+q1,k2+q2) G_{cb}(k1,k2)
    using broadcasting
    Parameters:
    G : np.ndarray.shape = (Lq1,Lq2,a,c) which are ky momentum, kx momentum, orbital_out, orbital_in
    Returns:
    z : np.ndarray.shape = (Lq1,Lq2,a,b) which are qy momentum, qx momentum, orbital_out, orbital_in
    """
    (Lq1,Lq2,_,_) = G.shape

    # create index arrays
    k1 = np.arange(Lq1)[:,None,None,None]
    k2 = np.arange(Lq2)[None,:,None,None]
    q1 = np.arange(Lq1)[None,None,:,None]
    q2 = np.arange(Lq2)[None,None,None,:]
    # compute shifted indices
    idx1 = (k1 + q1) % Lq1
    idx2 = (k2 + q2) % Lq2
    # broadcast multiplication, contract
    z = np.einsum('klqrac,klqrcb->qrab', G[idx1, idx2], G[k1,k2])/Lq1/Lq2
    
    return z


def cross_correlation_2loops(G: np.ndarray) -> np.ndarray:
    """Only for testing.
    Computes z_ab(q1,q2) = sum_{k1,k2,c} G_{ac}(k1+q1,k2+q2) G_{cb}(k1,k2)
    using 2 loops + np.roll
    Parameters:
    G : np.ndarray.shape = (Lq1,Lq2,a,c) which are ky momentum, kx momentum, orbital_out, orbital_in
    Returns:
    z : np.ndarray.shape = (Lq1,Lq2,a,b) which are qy momentum, qx momentum, orbital_out, orbital_in
    """
    (Lq1,Lq2,ndimH,ndimH) = G.shape

    s=np.zeros((Lq1,Lq2,Lq1,Lq2,ndimH,ndimH),dtype=complex)
    for q1 in range(Lq1):
        for q2 in range(Lq2):
            G_shifted = np.roll(np.roll(G, shift=-q1, axis=0), shift=-q2, axis=1)
            s[q1,q2] = np.einsum('klac,klcb->klab', G_shifted, G)

    z = np.sum(s,axis=(2,3))/Lq1/Lq2

    return z


def cross_correlation_4loops(G: np.ndarray) -> np.ndarray:
    """Only for testing.
    Computes z_ab(q1,q2) = sum_{k1,k2,c} G_{ac}(k1+q1,k2+q2) G_{cb}(k1,k2)
    using 4 loops
    Parameters:
    G : np.ndarray.shape = (Lq1,Lq2,a,c) which are ky momentum, kx momentum, orbital_out, orbital_in
    Returns:
    z : np.ndarray.shape = (Lq1,Lq2,a,b) which are qy momentum, qx momentum, orbital_out, orbital_in
    """
    (Lq1,Lq2,ndimH,ndimH) = G.shape

    s=np.zeros((Lq1,Lq2,Lq1,Lq2,ndimH,ndimH),dtype=complex)
    for k1 in range(G.shape[0]):
        for k2 in range(G.shape[1]):
            for q1 in range(G.shape[0]):
                for q2 in range(G.shape[1]):
                    s[q1,q2,k1,k2] = np.einsum('ac,cb->ab', G[(k1+q1)%Lq1,(k2+q2)%Lq2], G[k1,k2])

    z = np.sum(s,axis=(2,3))/Lq1/Lq2

    return z