#TODO: add tests for this module






import numpy as np
from numpy import pi,cos,sin,exp
import matplotlib.pylab as plt

#this is a dublicate


def find_phase_boundary(order_parameter,xs,ys,eps=1e-4):
    """Determines the points at which the order_parameter (.shape=(X,Y)) becomes non-zero. Coordinates of this array are xs (.shape=(X)), ys (.shape=(Y))"""

    coords = np.swapaxes(np.meshgrid(xs,ys,indexing='ij'),0,-1)

    order_parameter = 1*(np.abs(order_parameter)>eps) #True if inside the phase

    #check where phase boundary is
    jump0 = np.array(np.abs(order_parameter[1:]-order_parameter[:-1]),dtype=bool) #take the derivative
    coordinates0 = (coords[1:]+coords[:-1])/2 #compute the x,y coordinates of the derivative
    jump1 = np.array(np.abs(order_parameter[:,1:]-order_parameter[:,:-1]),dtype=bool)
    coordinates1 = (coords[:,1:]+coords[:,:-1])/2
    #list of x,y values phase boundary
    phase_boundary_xy = np.concatenate((coordinates0[jump0],coordinates1[jump1])) #.shape=(.,2)
    #now sort the list acording to distance from starting point
    jstart = np.argmin(np.min(phase_boundary_xy,axis=1))#select starting element, the one with smallest sup-norm
    norms = np.linalg.norm(phase_boundary_xy-phase_boundary_xy[np.newaxis,jstart,:],axis=1)
    asort = np.argsort(norms) #sort the list by its norm
    phase_boundary = phase_boundary_xy[asort].T #.shape=(2,.) sorted by x-value
    
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



#######################################
#intersections

def intersection_2lines(Q1,Q2,P1,P2):
    """Finds intersection of 2 lines defined by g_P from P1 to P2 and g_Q from Q1 to Q2
    Q1,Q2,P1,P2 are 2d"""
    #define g_P: (P2-P1)*s + P1
    #define g_Q: (Q2-Q1)*t + Q1
    s = ((Q2[1]-Q1[1])*(P1[0]-Q1[0]) - (Q2[0]-Q1[0])*(P1[1]-Q1[1])) / ((Q2[0]-Q1[0])*(P2[1]-P1[1]) - (Q2[1]-Q1[1])*(P2[0]-P1[0]))
    t = ((P2[0]-P1[0])*s + P1[0] - Q1[0]) / (Q2[0]-Q1[0])
    if 0<s<=1 and 0<t<=1: #intersection between each of the 2 points
        intersec_point = (P2-P1)*s + P1
        return True,intersec_point
    else:
        return False,[np.NaN, np.NaN]

    
def intersection_curves(points1,points2,return_intersec_indices=False):
    """Calculates all intersections of 2 curves"""
    intersec_points = []
    intersec_segment_index1 = []
    intersec_segment_index2 = []
    for i1 in range(len(points1)-1):
        for i2 in range(len(points2)-1):
            exists, S = intersection_2lines(points1[i1],points1[i1+1],points2[i2],points2[i2+1])
            if exists:
                intersec_points.append(S)
                intersec_segment_index1.append(i1)
                intersec_segment_index2.append(i2)
    if return_intersec_indices:
        return intersec_points, intersec_segment_index1, intersec_segment_index2
    else:
        return intersec_points

    
def split_intersected_curves_in_MinMax(points1,points2,return_bubbles=False):
    """For two curves defined by (2,.) arrays. Return the upper and lower cruve including intersections between them .shape(2,.)"""
    points1 = points1.T
    points2 = points2.T
    Ss, is1, is2 = intersection_curves(points1,points2, return_intersec_indices=True)
    
    #add starting values to it
    is1.insert(0, -1)
    is2.insert(0, -1)
    
    points_min = [[0,0]]
    points_max = [[0,0]]
    intersections = []
    jmin = np.argmin([points1[is1[1],1],points2[is2[1],1]]) #check which is the lower curve in the first segment
    for i in range(0,len(is1)-1):
        if jmin==0:
            points_min = np.concatenate((points_min,points1[is1[i]+1:is1[i+1]+1],[Ss[i]]))
            points_max = np.concatenate((points_max,points2[is2[i]+1:is2[i+1]+1],[Ss[i]]))
        if jmin==1:
            points_min = np.concatenate((points_min,points2[is2[i]+1:is2[i+1]+1],[Ss[i]]))
            points_max = np.concatenate((points_max,points1[is1[i]+1:is1[i+1]+1],[Ss[i]]))
        intersections.append([len(points_min)-1,len(points_max)-1]) #saves for each intersection at which index in points_min/max it occurs
        jmin = (jmin+1)%2 #now the other curve is the lower curve
        
    #add the last points
    if jmin==0:
        points_min = np.concatenate((points_min,points1[is1[-1]+1:]))
        points_max = np.concatenate((points_max,points2[is2[-1]+1:]))
    if jmin==1:
        points_min = np.concatenate((points_min,points2[is2[-1]+1:]))
        points_max = np.concatenate((points_max,points1[is1[-1]+1:]))
            
    
    if return_bubbles==False:
        return points_min[1:].T,points_max[1:].T
    else: #return_bubbles==True
        intersections = np.array(intersections)
        
        bubbles = []
        #first bubble (open)
        bubbles.append(np.concatenate((points_min[1:intersections[0,0]],points_max[1:intersections[0,1]+1][::-1])).T)
        #closed bubbles
        for j in range(1,len(intersections)):
                bubbles.append(np.concatenate((points_min[intersections[j-1,0]:intersections[j,0]],points_max[intersections[j-1,1]+1:intersections[j,1]+1][::-1])).T)
        #last bubble (open)
        bubbles.append(np.concatenate((points_min[intersections[-1,0]:][::-1],points_max[intersections[-1,1]+1:])).T)
        return points_min[1:].T,points_max[1:].T, bubbles
        


def split_intersected_curves_1intersec(points1,points2):
    """For two curves defined by (2,.) arrays. Return the upper and lower cruve including intersections between them .shape(2,.)"""
    points1 = points1.T
    points2 = points2.T
    S, i1, i2 = intersection_curves(points1,points2, return_intersec_indices=True)
    S = S[0]; i1 = i1[0]; i2 = i2[0]
    
    jmin = np.argmin([points1[i1,1],points2[i2,1]]) #check which is the lower curve in the last segment
    
    if jmin==0:
        points_min = np.concatenate((points1[0:i1+1],[S],points2[i2+1:]))
        points_max = np.concatenate((points2[0:i2+1],[S],points1[i1+1:]))
        points_upper_right = np.concatenate((points1[i1+1:][::-1],[S],points2[i2+1:]))
    else:
        points_min = np.concatenate((points2[0:i2+1],[S],points1[i1+1:]))
        points_max = np.concatenate((points1[0:i1+1],[S],points2[i2+1:]))
        points_upper_right = np.concatenate((points2[i2+1:][::-1],[S],points1[i1+1:]))
    
    return points_min.T,points_max.T, points_upper_right.T
