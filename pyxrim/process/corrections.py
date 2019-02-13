# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:33:23 2015

@author: nouamanelaanait
"""

import warnings, os
import numpy as np
import h5py
import multiprocess as mp
from skimage.morphology import dilation, disk
from scipy.ndimage import gaussian_filter
import cv2

class Corrector(object):
    ''' This is an Object used to contain a data set and has methods to do image correction 
        such as background subtraction, normalization, denoising, etc. 
        Basically, methods that manipulate the intensity values pixel by pixel.
    '''
    
    def __init__(self):
        self.__init__
        self.data = []
        self.proc_data = []
        
    def loadData(self, dataset):
        ''' This is a Method that loads h5 Dataset to be corrected.
            input: h5 dataset
        '''
        if not isinstance(dataset, h5py.Dataset):
            warnings.warn( 'Error: Data must be an h5 Dataset object'   )     
        else:
            self.data = dataset
            
    def loadprocData(self, dataset):
        ''' This is a Method that loads processed data to be corrected.
            input: h5 dataset or numpy ndarray
        '''
        self.proc_data = dataset        
            
    def clearData(self):
        ''' This is a Method to clear the data from the object.
        '''
        del self.data
        self.data = []  
        
    def getData(self):
        ''' This is a Method that returns loaded h5 Dataset.
            output: h5 dataset
        '''
        return self.data

    def bkgSubtract(self, impRead, impDark, impFlat = None):
        ''' This is a Method to correct image for READ, DARK noise of a ccd camera. 
        Also normalizes with respect to FlatField if provided.
        Input:
            impRead: np.ndarray
            impDark: np.ndarray
            impFlat = None, flatfield Image
            The size of the above images must match the raw image.
        '''
        
        dset = self.data
        att = dset.attrs
        exposure = att['seconds']
        impList = []
        
        if impFlat is None:
            try:
                for t, raw in zip(exposure, dset):
                    imp = raw - impRead - impDark*t
                    imp[imp<0] = 1
                    impList.append(imp)
                
                corrstack = np.dstack([impList]) 
		self.proc_data = corrstack
            except ValueError:
                warnings.warn('Error: Correction Files might not have the same size as the Image.')
                
        else:
            try: 
                for t, raw in zip(exposure, dset):
                    imp = (raw - impRead - impDark*t)/impFlat
                    imp[imp<0] = 1
                    impList.append(imp)
                    
                corrstack = np.dstack([impList])   
		self.proc_data = corrstack                    
            except ValueError:
                warnings.warn('Error: Correction Files might not have the same size as the Image.')
#        self.proc_data = corrstack
        return corrstack
            
    def normalize(self, monitor = None, trans = 'Trans', time = 'seconds', calibration = 1.0, use_processed = False):
        ''' Normalize the counts in the image by filters, exposure/or as monitor (e.g ion chamber) 
        and converts counts to photons if calibration is provided.
        Input:        
            calibration = float.
            monitor = string for the monitor attribute.
            use_processed = bool, if True uses the latest corrected data.
        '''
        if use_processed:
            data = self.proc_data
        else:
            data = self.data
        dset = self.data
        att = dset.attrs
        exposure = att[time]
        transmission = att[trans]
        impList = []
        if monitor is None:
            try: 
                for raw,t,trans in zip(data, exposure, transmission):
                    imp = calibration * raw / trans /t
                    impList.append(imp)                   
                normstack = np.dstack([impList])
		self.proc_data = normstack
            except ValueError:
                warnings.warn('Error: Couldnt broadcast seconds, trans and dset together')
        
        else:
            mon = att[monitor]
            try: 
                for raw,trans,m in zip(data,transmission,mon):
                    imp = calibration * raw / trans / m 
                    impList.append(imp) 
                normstack = np.dstack([impList])
		self.proc_data = normstack
            except ValueError:
                warnings.warn('Error: Couldnt broadcast trans, monitor, and dset  arrays together.')       
        return normstack
     
    def flatField(self, processes , sigmaMorph = 100, radiusMorph = 25, winLP = 7, sigmaBlur = 100,
                  use_processed = False, method = 'morphology'):
        ''' Find illumination function of the data and divide out the image for flat-field correction.
        
        Input:
            processes = int, number of processors to use.
            sigmaMorph = float, sigma of gaussian filter.
            radiusMorph = int, radius of structuring element.
            winLP = int, diameter of low-spatial frequency to cut out.
            use_processed = bool, if True uses the latest corrected data.
            method = 'morphology': A combination of morphological Filter (Dilation) and Gaussian (Blur) filter. 
            method = 'lowpass filter': gaussian lowpass filter applied directly to image.
           
        Output:
            stack of flat-field corrected images.
        '''
        if use_processed:
           data = self.proc_data
        else:
           data = self.data  
            
        def __flattenbyMorpho(imp):
            dilated = cv2.dilate(imp, disk(radiusMorph))
            #dilated = dilation(np.log10(imp), selem= disk(radius))
            illum= gaussian_filter(dilated, sigmaMorph)
            proc = imp*1.0/illum
            return proc 
            
        def __flattenbyLPF(imp):
            #Fourier transform
            FT = np.fft.fft2(imp)
            FT_shift = np.fft.fftshift(FT)
            #Cut-out low frequency in FT by gaussian LP filter
            row, col = imp.shape
            cen_row,cen_col = row/2 , col/2
            arr = np.ones(imp.shape)
            arr[cen_row-winLP/2:cen_row+winLP/2, cen_col-winLP/2:cen_col+winLP/2] = 0
            sigma = winLP
            LPfilt = gaussian_filter(arr, sigma/2, truncate = 5.)
            LPfilt[LPfilt < 0.9999] = 0.
            LPfilt = gaussian_filter(LPfilt, sigma, truncate = 5.)
            LPfilt[cen_row,cen_col]=1
            # Apply LP Filter and Inverse Fourier transform
            FT_ishift = np.fft.ifftshift(FT_shift*LPfilt)
            iFT = np.fft.ifft2(FT_ishift)
            proc = np.abs(iFT)
            return proc
            
        # start pool of workers
        print('launching %i kernels...'%(processes))
        pool = mp.Pool(processes)
        tasks = [(imp) for imp in data]
        chunk = int(data.shape[0]/processes)
        if method == 'morphology':
            jobs = pool.imap(__flattenbyMorpho, tasks, chunksize = chunk) 
        elif method == 'lowpass filter':
            jobs = pool.imap(__flattenbyLPF, tasks, chunksize = chunk)
                
        # get images from different processes
        results =[]
        print('Extracting Flattened Images...')
        try:
            for j in jobs:
                results.append(j)
        except ValueError:
            warnings.warn('Error: There appears to be a problem with the processing')
        
        
        # pack all images into 3d array
        flatstack = np.array([imp for imp in results])
        
        # close the pool
        print('Closing down the kernels... \n')
        pool.close() 
#        self.proc_data = flatstack
        return flatstack  
         
         
         
         
