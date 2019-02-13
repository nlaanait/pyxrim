# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:40:43 2015

@author: nouamanelaanait
"""

import os
import cv2
import numpy as np
from skimage.feature import match_descriptors, register_translation
from skimage.transform import warp, SimilarityTransform
from skimage.measure import ransac
import h5py
import multiprocess as mp
from misc import *

class geoTransformer(object):
    ''' This object contains methods to perform geometric transformations on 
    a sequence of images. Some of the capabilities are:
    + Homography by feature extraction.
    + Intensity-based image registration.
    + Projection Correction.
    '''
    def __init__(self):
        self.__init__
        self.data = []
        self.features = []
#        self.lib = lib
        
    def clearData(self):
        ''' This is a Method to clear the data from the object.
        '''
        del self.data
        self.data = []        
        
    def loadData(self, dataset):
        ''' This is a Method that loads the data to be geometrically-Transformed.
            input: h5 dataset, or 3D np.ndarray 
        '''
        
        self.data = dataset
    
    def loadFeatures(self, features):
        ''' This is a Method that loads features to be used for homography etc ...
        input: 
            features : [keypoints, descriptors].
                These can come from FeatureExtractor.getFeatures() or elsewhere.
                The format is :
                    keypoints = [np.ndarray([y_position, x_position])]
                    descriptors = [np.ndarray()]
        '''
        self.features = features    
    
    def matchFeatures(self, **kwargs):
        ''' This is a Method that computes similarity between keypoints based on their
        descriptors. Currently only skimage.feature.match_descriptors is implemented.
        In the future will need to add opencv2.matchers.
        Input:
            processors: int, optional
                    Number of processors to use, default = 1.
            lib: string, optional
                    CV Library to use (e.g. OpenCV, skimage).Currently only skimage works.
                    default, skimage.
            maximum_distance: int, optional
                    maximum_distance (int) of misalignment, default = infinity.
                    Used to filter the matches before optimizing the transformation.
        Output:
            Matches.
        '''
        desc = self.features[-1]
        keypts = self.features[0]
        processes = kwargs.get('processors', 1)
        maxDis = kwargs.get('maximum_distance', np.infty)
        lib = kwargs.get('lib', 'skimage' ) 
        
        
        def match(desc):
            desc1, desc2 = desc[0], desc[1]
            matches = match_descriptors(desc1, desc2, cross_check=True)
            return matches
        
        # start pool of workers
        pool = mp.Pool(processes)
        print('launching %i kernels...'%(processes))
        
        tasks = [ (desc1, desc2) for desc1, desc2 in zip(desc[:],desc[1:]) ]
        chunk = int(len(desc)/processes)
        jobs = pool.imap(match, tasks, chunksize = chunk) 
        
        # get matches
        print('Extracting Matches From the Descriptors...')
         
        matches =[]
        for j in jobs:
            matches.append(j)
            
        # close the pool
        print('Closing down the kernels...\n')
        pool.close() 
        
        # impose maximum_distance misalignment constraints on matches
        filt_matches = []
        for match, key1, key2 in zip(matches, keypts[:],keypts[1:]):
            filteredMask = euclidMatch(match, key1, key2, maxDis) 
            filt_matches.append(match[filteredMask])
            
        
        return matches, filt_matches
        
        
    def findTransformation(self, transform, matches, processes, **kwargs):
        ''' This is a Method that finds the optimal transformation between two images
        given matching features using a random sample consensus.
            Input:
                transform: skimage.transform object
                matches (list): matches found through match_features method.
                processors: Number of processors to use. 
                **kwargs are passed to skimage.transform.ransac
                
            Output:
                Transformations.
        '''

        keypts = self.features[0]
        
        def optimization(Pts):
            robustTrans, inliers = ransac((Pts[0], Pts[1]), transform, **kwargs)
            output = [robustTrans, inliers]
            return output
            
         # start pool of workers
        print('launching %i kernels...'%(processes))
        pool = mp.Pool(processes) 
        tasks = [ (key1[match[:, 0]], key2[match[:, 1]])
                    for match, key1, key2 in zip(matches,keypts[:],keypts[1:]) ]
        chunk = int(len(keypts)/processes)
        jobs = pool.imap(optimization, tasks, chunksize = chunk) 
        
        # get Transforms and inlier matches
        transforms, trueMatches =[], []
        print('Extracting Inlier Matches with RANSAC...')
        try:        
            for j in jobs:
                transforms.append(j[0])
                trueMatches.append(j[1])
        except np.linalg.LinAlgError:
            pass
            
        # close the pool
        pool.close() 
        print('Closing down the kernels...\n')
        
        return transforms, trueMatches
        
        
    def applyTransformation(self, transforms, **kwargs):
        ''' This is the method that takes the list of transformation found by findTransformation
         and applies them to the data set.
         
         Input:
             transforms: (list of skimage.GeoemetricTransform objects). 
                     The objects must be inititated with the desired parameters.
             transformation: string, optional.
                     The type of geometric transformation to use (i.e. translation, rigid, etc..)
                     Currently, only translation is implemented. 
                     default, translation.
             origin: int, optional
                     The position in the data to take as origin, i.e. don't transform.
                     default, center image in the stack.
             processors: int, optional
                    Number of processors to use, default = 1.
                    Currently,only one processor is used.
                    
        Output:
            Transformed images, transformations            
             
        '''
        dic = ['processors','origin','transformation']
        for key in kwargs.keys():
            if key not in dic:
                print('%s is not a parameter of this function' %(str(key)))
                
        processes = kwargs.get('processors', 1)
        origin = kwargs.get('origin', int(self.data.shape[0]/2))
        transformation = kwargs.get('transformation','translation')
        
        dset = self.data        
        # For now restricting this to just translation... Straightforward to generalize to other transform objects.
        if transformation == 'translation':
            
            YTrans = np.array([trans.translation[0] for trans in transforms])
            XTrans = np.array([trans.translation[1] for trans in transforms])
            chainL = []
            for y, x in zip(range(0,YTrans.size+1), range(0,XTrans.size+1)):
                if y < origin:
                    ychain = -np.sum(YTrans[y:origin])
                    xchain = -np.sum(XTrans[x:origin])
            
                elif y > origin:
                    ychain = np.sum(YTrans[origin:y])
                    xchain = np.sum(XTrans[origin:x])
                else:
                    ychain = 0
                    xchain = 0
                
                chainL.append([xchain, ychain])
            
            chainTransforms = []
            for params in  chainL:
                T = TranslationTransform(translation = params)
                chainTransforms.append(T)
                
        # Just need a single function that does boths        
        if transformation == 'rotation':
            
            rotTrans = np.array([trans.rotation for trans in transforms])
            YTrans = np.array([trans.translation[0] for trans in transforms])
            XTrans = np.array([trans.translation[1] for trans in transforms])
            chainL = []
            for x in range(0,rotTrans.size+1):
                if x < origin:
                    rotchain = -np.sum(rotTrans[x:origin])
                    ychain = -np.sum(YTrans[x:origin])
                    xchain = -np.sum(XTrans[x:origin])
                    
                elif x > origin:
                    rotchain = np.sum(rotTrans[origin:x])
                    ychain = np.sum(YTrans[origin:x])
                    xchain = np.sum(XTrans[origin:x])
                else:
                    rotchain = 0
                    ychain = 0
                    xchain = 0
             
                chainL.append([rotchain, xchain, ychain])
        
            chainTransforms = []
            for params in  chainL:
                T = SimilarityTransform(scale = 1.0, rotation = np.deg2rad(params[0]), translation = (params[1],params[2]))
#                T = SimilarityTransform(rotation = params, translation = (0,0))
                chainTransforms.append(T)
#        return chainTransforms
        # Use the chain transformations to transform the dataset
        output_shape = dset[0].shape
#        output_shape = (2048, 2048)
        def warping(datum):
            imp, transform  = datum[0], datum[1]
            transimp = warp(imp, inverse_map= transform, output_shape = output_shape, 
                            cval = 0, preserve_range = True)
            return transimp         
        
#          #start pool of workers
#         #somehow wrap function crashes when run in parallel! run sequentially for now.
#        pool = mp.Pool(processes)
#        print('launching %i kernels...'%(processes))
#        tasks = [ (imp, transform) for imp, transform in zip(dset, chainTransforms) ]
#        chunk = int(dset.shape[0]/processes)
#        jobs = pool.imap(warping, tasks, chunksize = 1)
#        #close the pool
#        pool.close()
#        print('Closing down the kernels... \n')       
#        
        # get transformed images and pack into 3d np.ndarray
        print('Transforming Images...')
        transImages = np.copy(dset[:])
               
        for imp, transform, itm in zip( transImages, chainTransforms, range(0,transImages.shape[0])):
            transimp = warping([imp, transform])
            transImages[itm] = transimp
            print('Image #%i'%(itm))
            

        return transImages, chainTransforms
        
    def correlationTransformation(self, **kwargs):
        ''' Uses Cross-correlation to find a translation between 2 images.
            Input:
                Processors: int, optional
                    Number of processors to use, default = 1.
                    
            Output:
                Transformations.  
        '''
        
        processes = kwargs.get('processors', 1)
                   
        pool = mp.Pool(processes)
        print('launching %i kernels...'%(processes))
 
        def register(images):
            imp1, imp2 = images[0], images[1]
            shifts, _, _ = register_translation(imp1,imp2)
            return shifts
            
        tasks = [ (imp1, imp2)
                    for imp1, imp2 in zip(self.data[:], self.data[1:]) ]
                        
        chunk = int((self.data.shape[0] - 1)/processes)
        jobs = pool.imap(register, tasks, chunksize = chunk) 
        
        # get Transforms and inlier matches
        results = []
        print('Extracting Translations')
        try:        
            for j in jobs:
                results.append(j)        
        except:
            print('Skipped Some Entry... dunno why!!')
            
        # close the pool
        pool.close() 
        
        return results
                
        
        
        
    