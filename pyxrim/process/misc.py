# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:35:43 2015

@author: nouamanelaanait
"""

import numpy as np
import math
from skimage.transform import ProjectiveTransform
import multiprocess as mp
import warnings
from scipy.signal import find_peaks_cwt, cwt

def pickle_keypoints(keypoints):
    ''' Function to pickle cv2.sift keypoint objects
    '''
    kpArray = np.array([])
    for point in keypoints:
        kpArray =np.append(kpArray, [point.pt[1],point.pt[0]])    
    kpArray =np.reshape(kpArray, (int(kpArray.size/2), 2))
    return kpArray
    
    

def euclidMatch(Matches, keypts1, keypts2, misalign):
    ''' Function that thresholds the matches, found from a comparison of
    their descriptors, by the maximum expected misalignment.
    '''
    filteredMatches = np.array([])
    deltaX =(keypts1[Matches[:,0],:][:,0]-keypts2[Matches[:,1],:][:,0])**2
    deltaY =(keypts1[Matches[:,0],:][:,1]-keypts2[Matches[:,1],:][:,1])**2
    dist = np.apply_along_axis(np.sqrt, 0, deltaX + deltaY)
    filteredMatches = np.where(dist[:] < misalign, True, False)
    return filteredMatches


def findPeaksArray(vectors, waveletWidths, processes = 4, **kwargs):
    '''
    Parallel version of find_peaks_cwt
    Input:
        vectors: 2d array, each column is a spectrum
        waveletWidths: 1d array of width peaks
        processes: # number of processes to use.
        kwargs: passed to scipy.signal.find_peaks_cwt
        
    Output:
        location of peaks
        
    '''
    
    # This is the function that will be mapped (in parallel) over the vectors 2d array
    def peaks(vector):
        peakIndices = find_peaks_cwt(vector, waveletWidths, **kwargs)
        return peakIndices        
        
    # start pool of workers
    print('launching %i kernels...'%(processes))
    pool = mp.Pool(processes)
    tasks = [(vector) for vector in vectors]
    chunk = int(vectors.shape[0]/processes)
    jobs = pool.imap(peaks, tasks, chunksize = chunk) 
    
    # get peaks from different processes
    results =[]
    print('Extracting Peaks...')
    try:
        for j in jobs:
            results.append(j)
    except ValueError:
        warnings.warn('Error: Something went wrong!!!')
        
    
    # pack all peaks into 2d array
    peaks = [itm for itm in results]
    
    # close the pool
    print('Closing down the kernels... \n')
    pool.close() 
    
    return peaks 




def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


def _center_and_normalize_points(points):
    """Center and normalize image points.

    The points are transformed in a two-step procedure that is expressed
    as a transformation matrix. The matrix of the resulting points is usually
    better conditioned than the matrix of the original points.

    Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    Parameters
    ----------
    points : (N, 2) array
        The coordinates of the image points.

    Returns
    -------
    matrix : (3, 3) array
        The transformation matrix to obtain the new points.
    new_points : (N, 2) array
        The transformed image points.

    """

    centroid = np.mean(points, axis=0)

    rms = math.sqrt(np.sum((points - centroid) ** 2) / points.shape[0])

    norm_factor = math.sqrt(2) / rms

    matrix = np.array([[norm_factor, 0, -norm_factor * centroid[0]],
                       [0, norm_factor, -norm_factor * centroid[1]],
                       [0, 0, 1]])

    pointsh = np.row_stack([points.T, np.ones((points.shape[0]),)])

    new_pointsh = np.dot(matrix, pointsh).T

    new_points = new_pointsh[:, :2]
    new_points[:, 0] /= new_pointsh[:, 2]
    new_points[:, 1] /= new_pointsh[:, 2]

    return matrix, new_points





class TranslationTransform(object):
    """ 2D translation using homogeneous representation:
    
    The transformation matrix is:
        [[1  1  tX]
         [1  1  tY]
         [0  0  1]]
         X: translation of x-axis.
         Y: translation of y-axis.
         
    Parameters:
    
    translation: (tX, tY) as a tuple.
    
    Attributes:
    
    params : (3, 3) array
        Homogeneous transformation matrix.
        
    """
    
    def __init__(self, matrix = None, translation = None):
        params = translation
        
        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix

        elif params:
            if translation is None:
                translation = (0., 0.)

            self.params = np.array([
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]
                ], dtype = 'float32')
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(3)
            
    def estimate(self, src, dst):
     #evaluate transformation matrix from src, dst
     # coordinates
        try:        
            xs = src[:, 0][0]
            ys = src[:, 1][1]
            xd = dst[:, 0][0]
            yd = dst[:, 1][1]
            S = np.array([[1., 0., xd-xs],
                          [0., 1., yd-ys],
                          [0., 0., 1.]
                          ],dtype = 'float32')
            self.params = S
            return True
        except IndexError:
            return False
            
    @property
    def _inv_matrix(self):
        inv_matrix = self.params
        inv_matrix[0:2,2] = - inv_matrix[0:2,2]
        return inv_matrix

    def _apply_mat(self, coords, matrix):
        coords = np.array(coords, copy=False, ndmin=2)

        x, y = np.transpose(coords)
        src = np.vstack((x, y, np.ones_like(x)))
        dst = np.dot(src.transpose(), matrix.transpose())

        # rescale to homogeneous coordinates
        dst[:, 0] /= dst[:, 2]
        dst[:, 1] /= dst[:, 2]

        return dst[:, :2]

    def __call__(self, coords):
        return self._apply_mat(coords, self.params)     
            
    def inverse(self, coords):
        ''' Apply inverse transformation.

        Parameters
        ----------
        coords : (N, 2) array
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.

        '''
        return self._apply_mat(coords, self._inv_matrix)
    def residuals(self, src, dst):
        """Determine residuals of transformed destination coordinates.

        For each transformed source coordinate the euclidean distance to the
        respective destination coordinate is determined.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        residuals : (N, ) array
            Residual for coordinate.

        """

        return np.sqrt(np.sum((self(src) - dst)**2, axis=1))
        
    @property
    def translation(self):
        return self.params[0:2, 2]       




        
class RigidTransform(object):
    """ 2D translation using homogeneous representation:
    
    The transformation matrix is:
        [[cos(theta)  -sin(theta)  tX]
         [sin(theta)  cos(theta)   tY]
         [0             0           1]]
         X: translation along x-axis.
         Y: translation along y-axis.
         theta: rotation angle in radians.
         
    Parameters:
    
    translation: (tX, tY) as a tuple.
    rotation: float in radians.
    
    Attributes:
    
    params : (3, 3) array
        Homogeneous transformation matrix.
        
    """
    
    def __init__(self, matrix = None, rotation = None, translation = None):
        params = any(param is not None
                     for param in (rotation, translation))
                         
        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix

        elif params:
            if translation is None:
                translation = (0, 0)
            if rotation is None:
                rotation = 0

            self.params = np.array([
                [math.cos(rotation), - math.sin(rotation), 0],
                [math.sin(rotation),   math.cos(rotation), 0],
                [                 0,                    0, 1]
            ])
            
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(3)
            
    def estimate(self, src, dst):
        """Set the transformation matrix with the explicit parameters.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        The transformation is defined as::

            X = a0 * x - b0 * y + a1
            Y = b0 * x + a0 * y + b1

        These equations can be transformed to the following form::

            0 = a0 * x - b0 * y + a1 - X
            0 = b0 * x + a0 * y + b1 - Y

        which exist for each set of corresponding points, so we have a set of
        N * 2 equations. The coefficients appear linearly so we can write
        A x = 0, where::

            A   = [[x 1 -y 0 -X]
                   [y 0  x 1 -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 b0 b1 c3]

        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        try:
            src_matrix, src = _center_and_normalize_points(src)
            dst_matrix, dst = _center_and_normalize_points(dst)
        except ZeroDivisionError:
            self.params = np.nan * np.empty((3, 3))
            return False

        xs = src[:, 0]
        ys = src[:, 1]
        xd = dst[:, 0]
        yd = dst[:, 1]
        rows = src.shape[0]

        # params: a0, a1, b0, b1
        A = np.zeros((rows * 2, 5))
        A[:rows, 0] = xs
        A[:rows, 2] = - ys
        A[:rows, 1] = 1
        A[rows:, 2] = xs
        A[rows:, 0] = ys
        A[rows:, 3] = 1
        A[:rows, 4] = xd
        A[rows:, 4] = yd

        _, _, V = np.linalg.svd(A)

        # solution is right singular vector that corresponds to smallest
        # singular value
        a0, a1, b0, b1 = - V[-1, :-1] / V[-1, -1]

        S = np.array([[a0, -b0, a1],
                      [b0,  a0, b1],
                      [ 0,   0,  1]])

        # De-center and de-normalize
        S = np.dot(np.linalg.inv(dst_matrix), np.dot(S, src_matrix))

        self.params = S

        return True


    def _apply_mat(self, coords, matrix):
        coords = np.array(coords, copy=False, ndmin=2)

        x, y = np.transpose(coords)
        src = np.vstack((x, y, np.ones_like(x)))
        dst = np.dot(src.transpose(), matrix.transpose())

        # rescale to homogeneous coordinates
        dst[:, 0] /= dst[:, 2]
        dst[:, 1] /= dst[:, 2]

        return dst[:, :2]

    def __call__(self, coords):
        return self._apply_mat(coords, self.params)     
            
    def inverse(self, coords):
        ''' Apply inverse transformation.

        Parameters
        ----------
        coords : (N, 2) array
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.

        '''
        return self._apply_mat(coords, self._inv_matrix)
    def residuals(self, src, dst):
        """Determine residuals of transformed destination coordinates.

        For each transformed source coordinate the euclidean distance to the
        respective destination coordinate is determined.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        residuals : (N, ) array
            Residual for coordinate.

        """

        return np.sqrt(np.sum((self(src) - dst)**2, axis=1))
     
     
    @property
    def _inv_matrix(self):
        return np.linalg.inv(self.params)
        
    @property
    def rotation(self):
        return math.atan2(self.params[1, 0], self.params[1, 1])

    @property
    def translation(self):
        return self.params[0:2, 2]    

def scaling(data, feature_range=(1,100)):
    '''
    Scaling data between feature_range.
    To retrieve original data: data = scaled*scaleFactor
    Input:
    data (np.array), feature_range(tuple)
    
    Output:
    
    scaled data(np.array), scalingfactor (np.array)
    '''
    #data = np.log(data)
    data_std = (data - data.min())/(data.max()- data.min())
    scaled = data_std * (feature_range[-1] - feature_range[0]) + feature_range[0]
    scaleFactor = data/scaled
    return scaled

def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray 
    
