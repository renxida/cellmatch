from skimage.io import imread, imsave, imshow
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
# depreciated
#from skimage.filters import sobel_h, sobel_v
from scipy.ndimage import gaussian_filter
from math import sqrt
import numpy as np
import os
#%%
sigma = 4.5
#%%
LINE_TYPE = 1 # or -1 for dark ridge (vallye) detection
os.chdir('/home/cedar/cellmatch')
img = imread('./a.jpg').astype(np.float64)
PI = np.pi
TAU = 2*np.pi
# sigma: 0.1113

#find greater eigenvalue that reflects maximum curvature direction.
# currently configured to find only ^ curves not v curves. Change key to 
# lambda x: x to find v curves
from itertools import product


def derivatives(img, sigma):
    # ALL RETURN VALUES ARE ARRAYS SAME SIZE AS img
    
    # convolve with 0th derivative along axis 0 and 1st derivative along axis 1 gives us x-gradient
    # x is axis 1, y is axis 0
    rx = gaussian_filter(img.astype(np.float64), sigma = sigma, order = (0,1))
    # vice versa
    ry = gaussian_filter(img.astype(np.float64), sigma = sigma, order = (1,0))
    # mode constant: used to handle image edges
    rxx, rxy, ryy = hessian_matrix(img, sigma = sigma, mode = 'nearest', cval = 0)
    return rx, ry, rxx, rxy, ryy

def slow_hessian_eigen(rxx, rxy, ryy):
    '''
    uses numpy linearalgebra to compute eigenvalues, eigenvectors
    don't use the slow one
    returns:
        ev -- eigenvalue with larger absolute value
        nx -- x-component of eigenvector for ev
        ny -- y-component of eigenvector for ev
        
    '''
    # prepare output variables of for loop
    nx, ny, ev = np.zeros_like(rxx), np.zeros_like(rxx), np.zeros_like(rxx)
    for r, c in product(range(512), repeat = 2):
        M = np.matrix([[rxx[r][c], rxy[r][c]], [rxy[r][c], ryy[r][c]]])
        vals, vects = np.linalg.eigh(M)
        # np.linalg.eigh sort eigenvalues in increasing order of face, non-absolute
        # value, so the most negative eigenvalue would be the one to use
        index_of_greater_eigenvalue = np.argmax(np.abs(vals))
        ev[r][c], nx[r][c], ny[r][c] = vals[index_of_greater_eigenvalue], vects[0, index_of_greater_eigenvalue], vects[1, index_of_greater_eigenvalue] # normalized eigenvector in direction perpendicular to line direction which is also the direction of maximum curvature
    return nx, ny, ev

def hessian_eigen(rxx, rxy, ryy):
    '''
    uses superpowered linear algebra to compute eigenvalues, eigenvectors
    problems? blame http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
    yeah. we use Harvard stuff yet we hate Harvard ;)
    returns:
        ev -- eigenvalue with larger absolute value
        nx -- x-component of eigenvector for ev
        ny -- y-component of eigenvector for ev
        
    '''
    # as directed by http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
    eigvals1, eigvals2= hessian_matrix_eigvals(rxx,rxy, ryy)
    majorev = np.where(np.abs(eigvals1)>np.abs(eigvals2),
                       eigvals1, eigvals2)
    # direction normal to line is direction of major eigenvector
    nonzero_nx = majorev - ryy
    nonzero_ny = rxy
    # normalize
    nlength = np.sqrt(nonzero_nx**2+nonzero_ny**2)
    if np.any(nlength==0):
        raise Exception('some eigenvectors have zero length')
    nonzero_nx = nonzero_nx/nlength
    nonzero_ny = nonzero_ny/nlength
    
    # when rxy is zero, use these instead
    zero_nx    = (rxx>rxy).astype(np.float64)
    zero_ny    = 2 - zero_nx
    
    nx = np.where(rxy == 0, zero_nx, nonzero_nx)
    ny = np.where(rxy == 0, zero_ny, nonzero_ny)
    return nx, ny, majorev

# test fast_slow equivalence
#fast = hessian_eigen(rxx, rxy, ryy)
#slow = hessian_eigen(rxx, rxy, ryy)
#for i in range(3):
## the abs is needed because eigenvectors can be 180 degrees from each other yet equivalent
#    print(np.sum(np.abs(fast[i])-np.abs(slow[i])))
    
def linepoints(derivatives, hessian_eigen):
    rx, ry, rxx, rxy, ryy = derivatives
    nx, ny, ev = hessian_eigen
    
    a = (rxx*nx**2+2*rxy*nx*ny+ryy*ny**2)
    b = -(rx*nx+ry*ny)
    t = b/a
    
    px = nx*t
    py = ny*t
    is_linepoint = (ev*LINE_TYPE<0) * (a!=0) * (np.abs(px)<0.5) * (np.abs(py)<0.5)
    return px, py, is_linepoint


rx, ry, rxx, rxy, ryy = derivatives(img, sigma = 3.5)
nx, ny, ev = hessian_eigen(rxx, rxy, ryy)
py, py, is_linepoint = linepoints(derivatives = (rx, ry, rxx, rxy, ryy), hessian_eigen = (nx, ny, ev))
imshow(is_linepoint)

#%%
#def hystereses_link
'''
line strength is equal to the second directional derivative of the image
in the direction normal to the line direction (i.e. in the direction of (nx,ny))
'''
# each angle corresponds to a block, and angles differing not too much gets the block
neighborhood_angles = np.arange(8)*np.pi/4
offset_x = np.around(np.cos(neighborhood_angles))
offset_y = np.around(np.sin(neighborhood_angles))
nhr = -offset_y.astype(np.int)
nhc =  offset_x.astype(np.int)
#neighborhood_offsets= np.array([
#        (+0, +1),
#        (-1, +1),
#        (-1, +0),
#        (-1, -1),
#        (+0, -1),
#        (+1, -1),
#        (+1, +0),
#        (+1, +1),])
#assert np.all(neighborhood_offsets == np.stack((nhr, nhc), axis = 1))
neighborhood_offsets = np.stack((nhr, nhc), axis = 1)
# TAU = 2PI
def find_neighborhood(head): #TODO: inspect
    head_orientation = orientation[head]
    angdiff_findneighbors = np.abs((head_angle-neighborhood_angles+PI)%(TAU)-PI)
    closest_index = np.argsort(angdiff_findneighbors)[0:3]
    neighborhood = neighborhood_offsets[closest_index] + head
    return neighborhood

def best_neighbor(head): #TODO: inspect
    '''
    takes: coordinate head, ndarray of coordinates neighbors
    returns: a number representing how well neighbor and head fit in a line.
    The greater the better.
    '''
    # find neighborhood
    neighborhood = find_neighborhood(head)
    # not sure what roles px, py are supposed to play. Just using pixel positions
    position_difference     = np.sqrt((head[0]-neighbors[:,0])**2 + (head[1]-neighbors[:,1])**2)
    
    head_orientation = orientation[head]
    neighborhood_orientations = orientation[tuple(neighborhood.T)]
    
    orientation_difference  = np.abs((head_orientation-neighborhood_orientations+PI)%TAU-PI)
    diff = position_difference + 1*orientation_difference
    return neighborhood[np.argmin(diff)]

def accumulate_points(seed, forward): #TODO? argumants may contain a threshold?
    '''
    returns a list of points added
    if   forward: normal of seed is on *right* of procession
    if ! forward: normal of seed is on *left*  of procession
    '''
    added_points = []
    head = seed
    while 1: # 1 = True
        angle = pangle[head]
        if not forward:
            angle = (angle - np.pi)%(2*np.pi) # -180deg and put back into range
        ntc = neighbors_to_consider(*head, angle)
        
        
# TODO: find out if we need to interpolate the 2nd derivative for px, py
# instead of pixel position. 
THRESHOLD_NEWLINE = 10 # threshold for derivative to initiate a new line at one point
# points to add have evs. point done don't have evs
remaining_point_line_strength= np.abs(is_linepoint*ev)
line_ids = np.full(ev.shape, -1, dtype = np.int64)
orientation = (np.arctan2(ny, nx) + (np.pi/2))%(2*np.pi) #parallel angle = normal angle + 90deg
npoints = np.count_nonzero(remaining_point_line_strength)
while npoints>0:
    seed = np.unravel_index(remaining_point_line_strength.argmax(), dims = remaining_point_line_strength.shape)
    if line_ids[seed]!=-1:
        raise ValueError('attempted to start line on point on another line')
    # set id, and mark as done.
    line_ids[seed] = npoints
    remaining_point_line_strength[seed] = 0
    npoints -= 1
    
    
    
    
    



#%%
