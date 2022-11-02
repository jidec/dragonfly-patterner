# -*- coding: utf-8 -*-
# Copyright (C) 2015 William R. Kuhn
"""
===============================================================================
IMAGE PREPROCESSING
===============================================================================
"""

import numpy as np
import cv2
import scipy.ndimage as nd
from skimage.morphology import convex_hull_image
from skimage.measure import regionprops, label
import joblib
from odomatic_utils import resize,pyramid,sliding_window
from skimage.filters import roberts
from skimage.color import rgb2gray

# Masking-related functions ===================================================

def michelson_constrast(arr):
    """Calculates Michelson's contrast for an array of intensity values:
        MC = (Imax-Imin)/(Imax+Imin)

    MC is returned as int between 0-255. Handles zero divisions.
    """
    mx,mn = float(arr.max()),float(arr.min()) # max and min values
    if mx==0.: # catch zero division, assumes mn is not negative
        return 0
    else:
        return int(((mx-mn)/(mx+mn))*255)

def michelson_constrast_transform(image):
    """Calculates Michelson's contrast on a single-channel intensity image.

    Parameters
    ----------
    image : array-like
        single-channel intensity image cast as np.uint8

    Returns
    -------
    michelson : list of arrays
        Michelson's contrast of `image` at scales 6x6 and 12x12-px as uint8
        arrays

    Sources
    -------
    pyramids: http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
    sliding windows: http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
    """
    # Get dimensions of original image
    h0,w0 = image.shape

    # Params for sliding window transformation
    winH,winW = (3,3) # window size; must be odd numbers
    stepSize = 1 # step size

    michelson = []

    # loop over the 1/2 & 1/4-sized images from image pyramid
    for i,resized in enumerate(pyramid(image, scale=2.,minSize=(100,100))):
        if i==0 or i>2:
            continue
        h,w = resized.shape[:2]
        out = np.zeros((h,w),dtype=np.uint8)

    	    # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            out[y,x] = michelson_constrast(window)

        michelson.append(out)

    # rescale transformed images back to size of `image`
    for i in range(len(michelson)):
        michelson[i] = cv2.resize(michelson[i],(w0,h0),interpolation=cv2.INTER_NEAREST)
    return michelson

def transparency_transform(image,bgr=False):
    """Returns a 6-channel transformation of an RGB image.

    Channels:
        (1) Cr (red-diff chroma) of YCrCb colorspace [1]
        (2) Cb (blue-diff chroma) of YCrCb colorspace [1]
        (3) S (saturation) of HSV colorspace [1]
        (4) MC6 : 6-px-window Michelson Contrast of Y (intens) of YCrCb [1,2]
        (5) MC12 : 12-px-window Michelson Contrast of Y [1,2]
        (6) E10 : Canny edge filtering + 10-px-radius blurring of Y

    Parameters
    ----------
    image : uint8 array, shape (h,w,3)
        Input image. Channels must be ordered RGB (not cv2's BGR)!
    bgr : bool, optional (default False)
        Whether to expect `image` channels to be in order BGR (as from
        cv2.imread()). Otherwise, channels assumed to be RGB

    Returns
    -------
    transformed_image : uint8 array, shape (h,w,6)
        Transformed image

    Sources:
    [1] Kompella, V.R., and Sturm, P. (2012). Collective-reward based approach
        for detection of semi-transparent objects in single images. Computer
        Vision and Image Understanding 116, 484–499.
    [2] https://en.wikipedia.org/wiki/Contrast_(vision)#Formula

    """
    if bgr: # convert image from BGR -> RGB
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    yy,cr,cb = cv2.split(cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb))
    h,s,v = cv2.split(cv2.cvtColor(image,cv2.COLOR_RGB2HSV))
    mc6,mc12 = michelson_constrast_transform(yy)

    e10 = cv2.blur(cv2.Canny(yy,100,200),(21,21))
    return np.dstack([cr,cb,s,mc6,mc12,e10])

def convex_hulls_image(binary_image):
    """Wrapper for skimage.morphology.convex_hull_image that applies that
    function to each object in the image, separately, returning a single bool
    image"""

    labeled = label(binary_image)
    indices = np.unique(labeled)[1:] # ignore first index (zero == background)
    true = np.ones(binary_image.shape,dtype=np.uint8)
    false = np.zeros(binary_image.shape,dtype=np.uint8)
    out = false.copy()
    for idx in indices:
        temp = np.where(labeled==idx,true,false)
        temp = convex_hull_image(temp) #Compute hull image
        out[temp] = True
    return out

def clear_border(binary_image, buffer_size=0, amt=0., bgval=0, in_place=False):
    """Remove objects connected to the image border.

    Parameters
    ----------
    labels : (M, N[, P]) array of int or bool
        Binary or labeled image.
    buffer_size : int, optional
        The width of the border examined.  By default, only objects
        that touch the outside of the image are removed.
    amt : float, optional
        Min threshold for how much of the border an object must span to be
        removed. By default, objects with any pixels in the border are removed.
    bgval : float or int, optional
        Cleared objects are set to this value.
    in_place : bool, optional
        Whether or not to manipulate the image array in-place.

    Returns
    -------
    out : (M, N[, P]) array
        Imaging data labels with cleared borders

    Adapted from `skimage.segmentation.clear_border()`
    """
    image = binary_image

    if any( ( buffer_size >= s for s in image.shape)):
        raise ValueError("buffer size may not be greater than image size")

    # create border mask using buffer_size
    borders = np.zeros_like(image, dtype=np.bool_)
    ext = buffer_size + 1
    slstart = slice(ext)
    slend   = slice(-ext, None)
    slices  = [slice(s) for s in image.shape]
    for d in range(image.ndim):
        slicedim = list(slices)
        slicedim[d] = slstart
        borders[slicedim] = True
        slicedim[d] = slend
        borders[slicedim] = True

    # Re-label, in case we are dealing with a binary image
    # and to get consistent labeling
    labels = label(image, background=0)
    number = np.max(labels) + 1

    bs = 1 if buffer_size==0 else buffer_size
    indices = np.arange(number + 1)
    proportions = [0] # set background proportion to zero
    for idx in indices[1:]:
        object_mask = np.where(labels==idx,np.ones(labels.shape),np.zeros(labels.shape))
        flat_borders = np.hstack((  object_mask[:bs],               # top
                                    object_mask[bs:-bs,-bs:].T,     # right
                                    object_mask[-bs:],              # bottom
                                    object_mask[bs:-bs,:bs].T))     # left

        flat_borders = np.any(flat_borders,axis=0) # collapse by columns, cast to bool
        # calc proportion of border cols/rows that are ones
        prop = len(flat_borders[flat_borders]) / float(len(flat_borders))
        proportions.append(prop)
    borders_indices = indices[np.array(proportions)>amt]

    # mask all label indices that are connected to borders
    label_mask = np.in1d(indices, borders_indices)
    # create mask for pixels to clear
    mask = label_mask[labels.ravel()].reshape(labels.shape)

    if not in_place:
        image = image.copy()

    # clear border pixels
    image[mask] = bgval

    return image

def _masker_v1(rgb_image,n_wings,scaler,clf,convex_hull=False,bgr=False):
    """Detect & mask wings in an image.

    Steps: (1) transform image to a get 6-len feature vector per pixel
           (2) apply trained classifier to predict whether each pixel is a wing
           (3) fill in holes in predicted mask
           (4) filter out all but largest `n_wings` regions in mask
           (5) optionally apply convex_hull_image to each kept region

    Parameters
    ----------
    rgb_image : uint8 array, shape (h,w,3)
        Input image. Channels must be ordered RGB (not cv2's BGR)!
    n_wings : int (1 or 2 typically)
        The number of wings to find and mask in `rgb_image`
    convex_hull : bool, optional (default False)
        Whether to return the convex hull of each object in the mask

    Returns
    -------
    mask : bool array, shape (h,w)
        Image mask, where `n_wings` objects (wings) are True and background
        pixels are False

    Raises RuntimeError if the number of recovered regions < n_wings.
    """

    batch_size_limit = 50000 # lower if raises MemoryError

    # Transform image to features for each pixel
    img_trans = transparency_transform(rgb_image)
    h,w,d = img_trans.shape # dims of transformed image

    # Flatten to shape (n_pixels,n_features)
    pixels = img_trans.reshape((h*w,d))

    # Predict whether each pixel is a wing
    batch_size = int(batch_size_limit/d)
    if len(pixels)>batch_size: # work in batches to prevent MemoryError
        # NOTE: uint8->float64 makes scaler transform memory intensive

        divs = int(len(pixels)/batch_size) # `divs`+1 batches will be used
        predicted = np.zeros((h*w),dtype=bool) # empty array to hold prediction

        for div in range(divs+1):
            if div<divs: # work with all but last batch
                batch = pixels[batch_size*div:batch_size*(div+1)]
                batch = batch.astype(np.float64) # cast to float for scaler
                batch = scaler.transform(batch)
                predicted[batch_size*div:batch_size*(div+1)] = clf.predict(batch)

            elif (len(pixels)%batch_size)>0: # last batch, if any remaining
                batch = pixels[batch_size*div:batch_size*(div+1)]
                batch = batch.astype(np.float64) # cast to float for scaler
                batch = scaler.transform(batch)
                predicted[-(len(pixels)%batch_size):] = clf.predict(batch)

    else: # if image is small, predict in a single go
        predicted = clf.predict(pixels)

    # Reshape back to image dims
    predicted_mask = predicted.reshape((h,w))

    # Do morphological hole filling
    filled_mask = nd.binary_fill_holes(predicted_mask) #Hole filling

    # Find regions in image and determine which to keep
    labeled_mask = label(filled_mask) #Label objects in bool image
    props = regionprops(labeled_mask) # region properties
    areas = [region['area'] for region in props]
    labels = np.unique(labeled_mask)
    labels = labels[1:] # drop region `0` (= image background)
    if len(labels)<n_wings: # catch if there aren't enough labeled regions
        raise RuntimeError('An insufficient number of objects was detected in image.')
    # keep only the `n_wings` regions that have the largest pixel area
    labels_to_keep = labels[np.argsort(areas)][-n_wings:]

    # Fill in empty mask with only the regions in labels_to_keep
    mask = np.zeros(labeled_mask.shape,dtype=int)
    for lbl in labels_to_keep:
        mask[labeled_mask==lbl] = lbl

    # apply convex hull to objects in image
    if convex_hull:
        mask = convex_hulls_image(mask)

        return mask.astype(bool)

def _masker_v4_2(rgb_image,n_wings,convex_hull=False,bgr=False):
    # reference: https://docs.opencv.org/3.3.1/d3/db4/tutorial_py_watershed.html

    ## detect edges in image
    edges = roberts(rgb2gray(rgb_image))
    edges *= 255./edges.max()
    edges = edges.astype(np.uint8)

    ## Distance transform edge map
    # Threshold & invert the image, so background is 0 and foreground is 1
    _,thresh = cv2.threshold(edges,0,255,cv2.THRESH_TRIANGLE+cv2.THRESH_BINARY_INV)
    # Distance transform: wing regions will be local minima
    dist = cv2.distanceTransform(thresh,cv2.DIST_L2,3)
    # Threshold again to catch those low-value regions
    _,thresh = cv2.threshold(dist,0.02*dist.max(),1.,cv2.THRESH_BINARY_INV)

    ## remove objects on border of image (possibly unnecessary)
    thresh = clear_border(thresh,buffer_size=25,amt=0.1)

    ## fill holes in remaining objects
    filled = nd.binary_fill_holes(thresh) #Hole filling

    ## remove small components (do this first!)
    labeled = label( filled.astype(bool) )
    areas = np.array([obj['area'] for obj in regionprops(labeled)])
    min_area = 3000 # min px area of objects to keep
    nix = np.arange(1,len(areas)+1)[areas<min_area]
    for i in nix:
        labeled[labeled==i] = 0

    thresh = labeled>0 # cast back to bool
    thresh = thresh.astype(np.uint8)*255
    ## compute watershed components
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    watershed = cv2.watershed(rgb_image,markers.copy())
    water_mask = np.zeros(markers.shape,dtype=np.uint8)
    water_mask[watershed > 1] = 255 # bg is 1, candidate objects are >1, contours are -1

    ## count remaining components
    labeled = label( water_mask.astype(bool) )
    nn = len(np.unique(labeled))-1 # no. remaining components

    if nn<n_wings: # not enough objects remain in image
        raise RuntimeError('An insufficient number of objects was detected in image.')
        #if len(np.unique(labeled))-1 >= nn+1, one of the wings was likely touching the image border

    ## keep only `n_wings` largest objects
    areas = np.array([obj['area'] for obj in regionprops(labeled)])
    keep = (np.argsort(areas)+1)[-n_wings:]
    largest_objs = np.zeros(thresh.shape,dtype=bool)
    for i in keep:
        largest_objs[labeled==i] = True

    # apply convex hull to objects in image
    if convex_hull:
        largest_objs = convex_hulls_image(largest_objs)

    return largest_objs.astype(bool)

def _masker_v4_3(rgb_image,n_wings,convex_hull=False,bgr=False):
    # reference: https://docs.opencv.org/3.3.1/d3/db4/tutorial_py_watershed.html

    ## detect edges in image
    edges = roberts(rgb2gray(rgb_image))
    edges *= 255./edges.max()
    edges = edges.astype(np.uint8)

    ## Distance transform edge map
    # Threshold & invert the image, so background is 0 and foreground is 1
    _,thresh = cv2.threshold(edges,0,255,cv2.THRESH_TRIANGLE+cv2.THRESH_BINARY_INV)
    # Distance transform: wing regions will be local minima
    dist = cv2.distanceTransform(thresh,cv2.DIST_L2,3)
    # Threshold again to catch those low-value regions
    _,thresh = cv2.threshold(dist,0.02*dist.max(),1.,cv2.THRESH_BINARY_INV)

    ## remove objects on border of image (possibly unnecessary)
    thresh = clear_border(thresh,buffer_size=25,amt=0.1)

    ## fill holes in remaining objects
    filled = nd.binary_fill_holes(thresh) #Hole filling

    ## remove small components (do this first!)
    labeled = label( filled.astype(bool) )
    areas = np.array([obj['area'] for obj in regionprops(labeled)])
    min_area = 3000 # min px area of objects to keep
    nix = np.arange(1,len(areas)+1)[areas<min_area]
    for i in nix:
        labeled[labeled==i] = 0

    thresh = labeled>0 # cast back to bool
    thresh = thresh.astype(np.uint8)*255
    ## compute watershed components
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=10)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    watershed = cv2.watershed(rgb_image,markers.copy())
    water_mask = np.zeros(markers.shape,dtype=np.uint8)
    water_mask[watershed > 1] = 255 # bg is 1, candidate objects are >1, contours are -1

    ## count remaining components
    labeled = label( water_mask.astype(bool) )
    nn = len(np.unique(labeled))-1 # no. remaining components

    if nn<n_wings: # not enough objects remain in image
        raise RuntimeError('An insufficient number of objects was detected in image.')
        #if len(np.unique(labeled))-1 >= nn+1, one of the wings was likely touching the image border

    ## keep only `n_wings` largest objects
    areas = np.array([obj['area'] for obj in regionprops(labeled)])
    keep = (np.argsort(areas)+1)[-n_wings:]
    largest_objs = np.zeros(thresh.shape,dtype=bool)
    for i in keep:
        largest_objs[labeled==i] = True

    # apply convex hull to objects in image
    if convex_hull:
        largest_objs = convex_hulls_image(largest_objs)

    return largest_objs.astype(bool)

# Functions supporting the Standardizer class =================================

def _check_image_mask_match(image,mask):
    return image.shape[:2]==mask.shape

def _load_model(filepath):
    return joblib.load(filepath)

def bounding_box(image,background='k'):
    """Get a bounding box for a mask or masked image. Image should only
    contain a single object. If image is color, its background color can
    be specified.

    Parameters
    ----------
    image : array, shape (h,w) or (h,w,3), dtype bool or uint8
        Input image. Should either be a bool mask or an image that has been
        masked, where the background is solid white or black.
    background : ('k'|'w'), optional (default 'k')
        Used if image is shape (h,w,3). Specifies background color of
        input image. 'k' for black or 'w' for white

    Returns
    -------
    output : tuple of ints
        Tuple of ints (min_row,min_col,max_row,max_col)

    Notes
    -----
    Background pixels in input image are presumed to be 0 or 0. and foreground
    pixels are >0. In this function, the first and last non-zero rows and
    columns are determined.
    """
    if image.ndim==3: # image is color
        g = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

        # map non-background pixels
        m = np.zeros(g.shape,dtype=bool)
        if background is 'k':
            m[g>0] = True
        elif background is 'w':
            m[g<255] = True
        else:
            raise ValueError('`background` value not understood.')

    elif image.dtype in (bool,np.bool): # image is a boolean mask
        m = image

    else:
        raise ValueError('`image` invalid.')

    # collapse `m` row-wise and column-wise
    rows = np.any(m,axis=1)
    cols = np.any(m,axis=0)

    # get indices for rows and cols that contain something non-background
    occupied_row_inds = np.arange(len(rows))[rows]
    occupied_col_inds = np.arange(len(cols))[cols]

    rmin = max(0,occupied_row_inds[0]-1)
    cmin = max(0,occupied_col_inds[0]-1)
    rmax = occupied_row_inds[-1]+1
    cmax = occupied_col_inds[-1]+1
    return (rmin,cmin,rmax,cmax)

def image_crop(image,bbox=None,background='k'):
    """Automatically crop a masked image, slicing away the mask. Or crop to
    provided bounding box.

    Parameters
    ----------
    image : ndarray
        Input mask or masked image.
    bbox : tuple, optional
        Allows a custom bounding box to be input. Must be in form
        (min_row,min_col,max_row,max_col), where all values are ints.
        Otherwise, `bounding_box()` is used to calculate `bbox`.
    background : ('k'|'w'), optional (default 'k')
        Used if image is shape (h,w,3). Specifies background color of
        input image. 'k' for black or 'w' for white.
        Passed to `bounding_box()`.

    Returns
    -------
    output : ndarray
        Image, cropped to bounding box.
    """

    #bbox should be in format (min_row, min_col, max_row, max_col)
    h,w = image.shape[:2]
    if bbox is None:
        bb = bounding_box(image,background=background) #Get bbox
    elif len(bbox)==4:
        bb = bbox
    else:
        raise RuntimeError('Image crop error: `bbox` form not understood.')

    # Slice image to bb (while preventing out-of-bounds slicing)
    return image[bb[0]:min(bb[2],h),bb[1]:min(bb[3],w)]

def sort_mask_regions(mask):
    """Splits mask by region, returning a sorted list of single-region masks.

    Parameters
    ----------
    mask : bool array, shape (h,w)
        Boolean mask containing 2 objects

    Returns
    -------
    output : list of bool arrays
        List of single-region Boolean masks, ordered from upper-most to
        lower-most object in `mask`
    """
    labeled_mask = label(mask) #Label objects in bool image
    labels = np.unique(labeled_mask)
    labels = labels[1:] # drop region `0` (= image background)

    # get centroids for each region (returned as ``(row,col)``)
    props = regionprops(labeled_mask) # region properties
    centroids = [region['centroid'] for region in props]
    # get y-value for each region's centroid
    ys = [y for y,x in centroids]
    # sort region labels by their y-value
    # (sorted upper-most region to lower-most)
    ordered_labels = labels[np.argsort(ys)]

    # use that order to make a sorted list of single-region masks
    output = []
    for lbl in ordered_labels:
        temp = np.zeros(mask.shape,dtype=bool) # create empty mask
        temp[labeled_mask==lbl] = True # fill specific region
        output.append(temp)
    return output

def apply_mask(image,mask,background='k'):
    """Safely masks an image.

    Parameters
    ----------
    image : unint8 array, shape (h,w,3)
        Input image.
    mask : bool array, shape (h,w)
        Boolean mask.
    background : ('k'|'w'), optional (default 'k')
        Desired background color for standardized square.
        'k' for black or 'w' for white.

    Returns
    -------
    masked_image : ndarray
        Masked image where black pixels in `mask` replace those corresponding
        pixels in `image`.
    """

    masked_image = cv2.bitwise_and(image,image,mask=mask.astype(np.uint8)*255)

    if background is 'k':
        return masked_image
    elif background is 'w':
        masked_image[~mask] = (255,255,255)
        return masked_image
    else:
        raise ValueError('`background` value not understood.')

def reorient_wing_image(image,cutoff=0.05,background='k'):
    """Reorients an object in an image so that its top side is horizontal.

    Parameters
    ----------
    image : uint8 array, shape (h,w,3)
        Input image (mask) or masked image containing a single object.
    cutoff : float, optional
        Object (i.e. wing) in mask is clipped, longitudinally on either side by
        this amount before reorientation. This reduces the effect of artifacts
        at the ends of the object.
    background : ('k'|'w'), optional (default 'k')
        Color of `image`'s background. 'k' for black or 'w' for white.

    Returns
    -------
    output : ndarray
        An image containing the object in *image* that has been reoriented and cropped.
    """

    # Use `cutoff` to get start and end columns
    w = image.shape[1]
    start = int(np.floor(w * cutoff))-1 #Starting col after clipping
    end = int(np.ceil(w * (1-cutoff))) #Ending col after clipping

    # Convert image to grayscale
    m = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    m = m.T # Transpose image so we can look at columns first, then rows

    if background is 'k': # if background is black
        bg = 0; func = max
    elif background is 'w': # if background is black
        bg = 255; func = min
    else:
        raise ValueError('`background` value not understood.')

    # Find optimal rotation angle
    tops = []
    for col in range(start,end,10): #For every 10th column...
        if func(m[col])!=bg: # if col contains non-background pixel
            templist = []
            # take index of first non-background pixel & append to tops
            for row,val in enumerate(m[col]):
                if val!=bg:
                    templist.append([col,row])
                else:
                    continue
                tops.append(templist[0])
        else:
            continue

    xs,ys = np.transpose(tops)

    slope,_ = np.polyfit(xs,ys,1) #Fit line to top coordinates
    slope = np.rad2deg(np.arctan(slope)) #Convert slope to degrees

    #Rotate image:
    rotated = nd.rotate(image,slope,mode='constant',cval=bg)
    rotated = image_crop(rotated,background=background)
    return rotated

def pad_and_stack(image_list,height=256,background='k'):
    """Pad images to `height`, then assemble with one above the other.

    Parameters
    ----------
    image_list : list of arrays
        List of two uint8 images, each of shape (h,w,3).
    height : int, optional
        Target height (in px) of each image after padding bottom edge with
        background pixels. Default is 256.
    background : ('k'|'w'), optional (default 'k')
        Desired background color for standardized square.
        'k' for black or 'w' for white.

    Returns
    -------
    output : uint8 array
        Image formed by padding the bottom of each image with background pixels
        and combining them so that the second image is below the first.

    Raises
    ------
    RuntimeError : if an image's width:height is less than 2. This typically
    catches masking errors.
    """
    if background is 'k': # if background is black
        bg = 0
    elif background is 'w': # if background is black
        bg = 255
    else:
        raise ValueError('`background` value not understood.')

    # pad the bottom of each image with background pixels
    padded_list = []
    for image in image_list:
        h,w = image.shape[:2]
        if h > height:
            raise RuntimeError('Mask aspect ratio is throwing off padding. Check mask.')
        pad_width = ((0,height-h),(0,0),(0,0))
        padded = np.pad(image,pad_width,mode='constant',constant_values=bg)
        padded_list.append(padded)

    return np.vstack(padded_list)


# STANDARDIZER CLASS============================================================

# By using this class, the scaler/clf models only have to be loaded once for
# standardizing multiple specimen images.
class Standardizer:
    """For transforming an image or pair of images into a standardized square
    image.

    Parameters
    ----------
    mask_method : str {`v1`|`v4_2`,`v4_3`}, default None
        Method for wing-masking. v1 for _masker_v1(), etc. None uses latest.
    scaler_fp : str|None, default None
        Filepath to pickle of pre-trained sklearn.preprocessing.StandardScaler()
        object. Needed for mask_method=`v1`
    clf_fp : str|None, default None
        Filepath to pickle of sklearn classifier, pre-trained to classify
        transformed pixels of `image` as either True (for foreground pixels)
        or False (for background pixels). Needed for mask_method=`v1`
    background : ('k'|'w'), optional (default 'k')
        Desired background color for standardized square.
        'k' for black or 'w' for white.
    convex_hull : bool, optional (default True)
        Whether to apply `skimage.morphology.convex_hull_image()` to wing
        masks before returning them. Passed to `mask_wings()`.
    bgr : bool, optional (default False)
        Whether to expect `image` channels to be in order BGR (as from
        cv2.imread()). Otherwise, channels assumed to be RGB. If bgr, returns
        BGR image, otherwise returns RGB.

    Attributes
    ----------
    scaler_ : sklearn.preprocessing.StandardScaler object
        Instance of a pre-trained StandardScaler object. Needed for _masker_v1()
    clf_ : sklearn classifier object
        Instance of a pre-trained classifier object that accepts pixel-wise
        features vectors from `transparency_transform(image)` and outputs True
        for pixels that are predicted to be from a wing and False for background
        pixels. Needed only for _masker_v1()

    Methods
    -------
    make_square : builds a standardized square from an image or pair of images

    Examples
    --------
    import autoID,cv2
    scaler_fp = 'path/to/file/scaler.pkl'
    clf_fp = 'path/to/file/clf.pkl'
    squarer = autoID.Standardizer(scaler_fp,clf_fp,background='k',bgr=True)
    img = cv2.imread('path/to/file/image_with_2_wings.tif') # read img as BGR
    square = squarer.make_square(img) # convert to standardized square
    """

    def __init__(self,mask_method=None,scaler_fp=None,clf_fp=None,background='k',convex_hull=False,
                 bgr=False):
        self.background     = background
        self.convex_hull    = convex_hull
        self.bgr            = bgr
        self.mask_method    = mask_method

        # load models (needed only for _masker_v1(), ignored otherwise)
        self.scaler_         = _load_model(scaler_fp) if scaler_fp else None
        self.clf_            = _load_model(clf_fp) if clf_fp else None

    def mask_wings(self,image,n_wings,method=None):
        """Wrapper for masking methods. Detect & mask wings in an image.

        Steps: (1) transform image to a get 6-len feature vector per pixel
               (2) apply trained classifier to predict whether each pixel is a wing
               (3) fill in holes in predicted mask
               (4) filter out all but largest `n_wings` regions in mask
               (5) optionally apply convex_hull_image to each kept region

        Parameters
        ----------
        image : uint8 array, shape (h,w,3)
            Input image. Channels must be ordered RGB (not cv2's BGR)!
        n_wings : int, optional (default 2)
            The number of wings to find and mask in `image`
        method : `v4_3`|`v4_2`|`v1`|`latest`|None, optional (default None)
            Which masking function to use. None & `latest` use latest function,
            or specify function by version number.

        Returns
        -------
        mask : bool array, shape (h,w)
            Image mask, where `n_wings` objects (wings) are True and background
            pixels are False

        Raises RuntimeError if the number of recovered regions < n_wings.
        """
        convex_hull = self.convex_hull

        if (method is None) or (method in ['latest','v4_3']): # use latest masking method
            return _masker_v4_3(image,n_wings,convex_hull)
        elif method=='v4_2': # use v4.2
            return _masker_v4_2(image,n_wings,convex_hull)
        elif method=='v1': # use version 1 masking method
            scaler      = self.scaler_
            clf         = self.clf_
            if clf is None or scaler is None: # check that clf & scaler are provided
                raise ValueError('Scaler and classifier must be provided for mask_method=`v1`.')
            return _masker_v1(image,n_wings,scaler,clf,convex_hull)
        else:
            raise ValueError('Masking method {!r} not recognized.')

    def make_square(self, image1, mask1=None, image2=None, mask2=None,
                    flip=False, switch=False):
        """Make a standardized square image from 1 two-winged image or 2
        one-winged images.

        Parameters
        ----------
        image1,image2 : uint8 array
            Input images, each containing 1 wing. Can be color or grayscale.
        mask1,mask2 : bool array, optional
            Allows custom Boolean masks to be submitted for image1 &/or image2.
            If not provided, `mask_wings()` is used to get mask.
        background : ('k'|'w'), optional (default 'k')
            Desired background color for standardized square.
            'k' for black or 'w' for white.
        bgr : bool, optional (default False)
            Whether channels of input images are BGR (as from `cv2.imread()`).
            Otherwise, images are assumed to be RGB. If bgr, returns
            BGR image, otherwise returns RGB. Passed to `mask_wings()`
        convex_hull : bool, optional (default True)
            Whether to apply `skimage.morphology.convex_hull_image()` to wing
            masks before returning them. Passed to `mask_wings()`.
        flip : bool,optional (default False)
            Option to flip images left-to-right.
        switch : bool,optional (default False)
            Option to switch the order of the first & second image.

        Returns
        -------
        output : uint8 array, shape (512,512,3)
            Standardized 512-px square image, where the fore- and hindwings are
            masked, reoriented, resized and placed in the upper and lower halves
            of the square, respectively. If `bgr=True`, image is returned as
            BGR, otherwise returned as RGB.

        Raises
        ------
        RuntimeError : if mask_wings() fails to find enough objects in image(s)
        """
        bg          = self.background

        # if wings are both contained in one image
        if image2 is None: # image1 contains 2 wings
            if mask1 is None: # get mask for image1
                mask1 = self.mask_wings(image1,n_wings=2)
            elif not _check_image_mask_match(image1,mask1):
                raise ValueError('`image1` and `mask1` are not the same shape.')

            sorted_masks = sort_mask_regions(mask1)
            if switch:
                sorted_masks = sorted_masks[::-1]
            masked_images = list(map(lambda y: apply_mask(image1,y,background=bg),
                                sorted_masks))

        # if wings are seperated between 2 images
        elif image2 is not None: # image1 and image2 each contain a wing
            if mask1 is None:
                mask1 = self.mask_wings(image1,n_wings=1)
            elif not _check_image_mask_match(image1,mask1):
                raise ValueError('`image1` and `mask1` are not the same shape.')

            if mask2 is None:
                mask2 = self.mask_wings(image2,n_wings=1)
            elif not _check_image_mask_match(image2,mask2):
                raise ValueError('`image2` and `mask2` are not the same shape.')

            if switch:
                images = [image2,image1]
                sorted_masks = [mask2,mask1]
            else:
                images = [image1,image2]
                sorted_masks = [mask1,mask2]
            masked_images = list(map(lambda x,y: apply_mask(x,y,background=bg),
                                images,sorted_masks))

        cropped_images = list(map(lambda x: image_crop(x,background=bg),
                             masked_images))
        rotated_images = list(map(lambda x: reorient_wing_image(x,background=bg),
                             cropped_images))
        resized_images = list(map(lambda x: resize(x,width=512),rotated_images))
        square = pad_and_stack(resized_images,background=bg)

        if flip:
            square = np.fliplr(square)

        if self.bgr:
            return square[:,:,::-1]
        else:
            return square
