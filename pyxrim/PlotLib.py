# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:23:05 2015

@author: nouamanelaanait
"""

import numpy as np
from matplotlib import pyplot as plt

def imageGallery(n_col, n_row, Title , images, **kwargs):
    ''' Plots a bunch of images.
    num := figure num
    n_col := # of columns
    n_rows := ...
    titleList := list of axes titles
    images := list of images
    **kwargs of plt.imshow
    '''
    fig, axes = plt.subplots(n_row, n_col, squeeze=True, figsize = (12, 6), sharex = True, sharey =True)
    plt.suptitle(Title, size=16)
    for ax, imp in zip(axes.flat, images):
        ax.imshow(imp, **kwargs)
        ax.axis('off')
        
def plotGallery(num, subtitles, xlist, ylist, n_col,n_row, Maintitle = '',**kwargs):
    '''
    Function for multiple (x,y) scatter plots.
    '''
    plt.figure(num, figsize=(5 * n_col, 5 * n_row))
    plt.suptitle(Maintitle, size=16)
    lst = [xlist, ylist, subtitles]
    #    for i, elem in zip(range(0,len(lst)+1),lst):
    for i in range(0,len(lst)+1):
        ax = plt.subplot(n_row, n_col, i+1 )
        plt.plot(lst[0][i], lst[1][i],**kwargs)
        ax.set_title(str(lst[2][i]))
        

def imageFeaturesGallery(n_col, n_row, Title, images ,keypts, **kwargs):
    ''' Plots a bunch of images and features .
    num := figure num
    n_col := # of columns
    n_rows := ...
    titleList := list of axes titles
    images := list of images
    **kwargs of plt.imshow
    '''
    fig, axes = plt.subplots(n_row, n_col, squeeze=True, figsize = (8, 8), sharex = True, sharey =True)
    plt.suptitle(Title, size=16)
#    plt.figure(num, figsize=( n_col,  n_row))
    if len(keypts) != 0 :    
        for ax, imp, key in zip(axes.flat, images, keypts):
            ax.imshow(imp)
            ax.scatter(key[:, 1], key[:, 0],marker ='D',c = 'k', s =15 )
            ax.axis('off')
    else:
        for ax, imp in zip(axes.flat, images):
            ax.imshow(imp, **kwargs)
            ax.axis('off')  
        
    plt.tight_layout(pad = 0.5, h_pad = 0.01, w_pad =0.01)

     
def plotMatches(ax, image1, image2, keypoints1, keypoints2, matches,
                 keypoints_color='k', matches_color=None, only_matches=False,
                 **kwargs):
    """Plot matched features.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matches and image are drawn in this ax.
    image1 : (N, M [, 3]) array
        First grayscale or color image.
    image2 : (N, M [, 3]) array
        Second grayscale or color image.
    keypoints1 : (K1, 2) array
        First keypoint coordinates as ``(row, col)``.
    keypoints2 : (K2, 2) array
        Second keypoint coordinates as ``(row, col)``.
    matches : (Q, 2) array
        Indices of corresponding matches in first and second set of
        descriptors, where ``matches[:, 0]`` denote the indices in the first
        and ``matches[:, 1]`` the indices in the second set of descriptors.
    keypoints_color : matplotlib color, optional
        Color for keypoint locations.
    matches_color : matplotlib color, optional
        Color for lines which connect keypoint matches. By default the
        color is chosen randomly.
    only_matches : bool, optional
        Whether to only plot matches and not plot the keypoint locations.
    **kwargs of imshow
    """

    image1 = img_as_float(image1)
    image2 = img_as_float(image2)

    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    image = np.concatenate([image1, image2], axis=1)

    offset = image1.shape

    if not only_matches:
        ax.scatter(keypoints1[:, 1], keypoints1[:, 0],
                   facecolors='none', edgecolors=keypoints_color)
        ax.scatter(keypoints2[:, 1] + offset[1], keypoints2[:, 0],
                   facecolors='none', edgecolors=keypoints_color)

    ax.imshow(image, cmap='jet', vmin= np.amin(image1), vmax=np.amax(image2)/6.5089)
#    ax.imshow(image, **kwargs)    
    ax.axis((0, 2 * offset[1], offset[0], 0))

    for i in range(matches.shape[0]):
        idx1 = matches[i, 0]
        idx2 = matches[i, 1]

        if matches_color is None:
            color = np.random.rand(3, 1)
        else:
            color = matches_color

        ax.plot((keypoints1[idx1, 1], keypoints2[idx2, 1] + offset[1]),
                (keypoints1[idx1, 0], keypoints2[idx2, 0]),
                '-', color=color)

def imageTile(data, padsize=1, padval=0, figsize=(12,12),**kwargs):
    '''
    Function to tile n-images into a single image   
    Input:  
    data: np.ndarray. Must be of dimension 3.  
    padsize: size in pixels of borders between different images. Default is 1.  
    padval: value by which to pad. Default is 0. 
    figsize: size of the figure passed to matplotlib.pyplot.figure. Default is (12,12).  
    **kwargs: extra arguments to be passed to pyplot.imshow() function.
    '''
   
    # force the number of images to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile all the images into a single image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.imshow(data,**kwargs)
    ax.axis('off')
       
