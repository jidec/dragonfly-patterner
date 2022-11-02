# -*- coding: utf-8 -*-
# Copyright (C) 2015 William R. Kuhn
"""
===============================================================================
UTILITIES
===============================================================================
"""
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2
import yaml

def read_config(filepath):
    with open(filepath,'r') as ymlfile:
        return yaml.load(ymlfile)

# GENERAL IMAGE FUNCTIONS ======================================================

def resize(image, width=None, height=None, prop=None, interp=cv2.INTER_LINEAR):
    """Convenient function for resizing an image specifying only desired
    width, height, or resized proportion of the resized image.

    Parameters
    ----------
    image : array, shape (h,w) or (h,w,d)
        Input image to resize
    width,height : int, optional (default None)
        If only `width` or `height` provided, it's treated as exact pixel
        size for that dimension of resized image & is used to calculate the
        other dimension proportionally to the input image.
        If both are provided, resized image will be shape (`width`,`height`).
        Returns unaltered input image if width=None,height=None and prop=None.
    prop : float, optional (default None)
        Resized image will be shape (orig_height * prop, orig_width * prop).
        Ignored if anything provided for width or height.
    interp : cv2 function
        OpenCV interpolation function used to resize image.
        e.g. cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_NEAREST

    Returns
    -------
    resized : array
        Resized image.

    Extended from `resize()` in Adrian Rosebrock's `imutils` package (v0.4.3)
    (http://www.pyimagesearch.com).
    """
    # Get image size
    h,w = image.shape[:2]

    # nothing is provided for width, height, or prop (all are None)
    if width is None and height is None and prop is None:
        return image

    # both width & height are provided
    if type(width) is int and type(height) is int:
        dim = (width,height)

    # only width is provided
    elif width is None and type(height) is int:
        # calculate the ratio of the height and construct dims
        r = height / float(h)
        dim = (int(w * r), height)

    # only height is provided
    elif height is None and type(width) is int:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # prop is provided
    elif prop is not None:
        dim = (int(w * prop), int(h * prop))
    # resize the image
    resized = cv2.resize(image, dim, interpolation=interp)

    # return the resized image
    return resized


def pyramid(image, scale=1.5, minSize=(30,30),interp=cv2.INTER_AREA):
    """Creates a generator for images in a pyramid.

    Parameters
    ----------
    image : array, shape (h,w,3)
        Input color image (RGB or BGR)
    scale : float, optional (default 1.5)
        How much to reduce the size of each successive image in the pyramid
    minSize : tuple, optional (default (30,30))
        Contrains the minimum image size (h,w) to be returned in the pyramid
    interp : function, optional (Default cv2.INTER_AREA)
        Interpolation function used to reduce image size

    Returns
    -------
    generator
        Returns original image, then image reduced by `scale`, etc. until next
        image is smaller than `minSize`

    Adapted from code at:
    http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
    """
	# yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / float(scale))
        image = resize(image, width=w,interp=interp)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(image, windowSize, stepSize=1):
    """Creates a generator of windows sliding across `image`.

    Parameters
    ----------
    image : array
        Input image of shape (h,w) or (h,w,d)
    windowSize : tuple
        Pixel size (h,w) of window to return
    stepSize : int, optional (default 1)
        Size of row-wise and column-wise steps to take between windows

    Returns
    -------
    generator
        Returns each successive window of size `windowSize` in row-wise and
        column-wise steps of `stepSize` until windows are exhausted

    Adapted from source:
    http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
    """
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def meanImage(imlist,dir=None,func=np.mean):
    """Find the mean (or median or stdev) of a set of same-sized images.

    Parameters
    ----------
    imlist : list
        List of str filenames of input images.
    dir : str, optional
        Optional directory string to prepend to each filename in imlist.
        Default is None, where imlist is used as is.
    func : np.mean, np.median, or np.std, optional
        Function applied, pixelwise, to images in imlist.
        Default is np.mean, where the mean image is returned.
        np.median and np.std return the median and standard deviation of imlist
        images, respectively.

    Returns
    -------
    imNew : array of dtype uint8
        Image array, where each pixel represents the mean (or median or stdev)
        for that pixel from all images in imlist.

    References
    ----------
    [1] http://stackoverflow.com/questions/17291455/how-to-get-an-average-picture-from-100-pictures-using-pil
    """
    if dir != None: #If directory is provided, append to each filename in imlist
        imlist = list(map(lambda x: os.path.join(dir,x),imlist))

    #Assuming all images are the same size, get dimensions of first image
    im0 = Image.open(imlist[0])
    w,h = im0.size
    if im0.mode == 'RGB':
        c0,c1,c2 = np.zeros((3,len(imlist),w*h))#Create empty array
        for i,im in enumerate(imlist):

            imarr = np.array(Image.open(im),dtype=np.float)#Open image, cast as float array
            i0,i1,i2 = np.transpose(imarr,(2,0,1))#Split channels

            #Flatten arrays
            i0=np.ravel(i0)
            i1=np.ravel(i1)
            i2=np.ravel(i2)

            #Append arrays to c0,c1,c2
            c0[i]=i0
            c1[i]=i1
            c2[i]=i2
        c0 = np.reshape(func(c0,axis=0),(w,h))
        c1 = np.reshape(func(c1,axis=0),(w,h))
        c2 = np.reshape(func(c2,axis=0),(w,h))
        imNew = np.dstack((c0,c1,c2))#Remerge channels

    else:
        c0 = np.zeros((len(imlist),w*h))#Create empty array
        for i,im in enumerate(imlist):
            imarr = np.array(Image.open(im),dtype=np.float)#Open image, cast as float array
            i0 = np.ravel(imarr)#Flatten array
            c0[i]=i0
        imNew = np.reshape(func(c0,axis=0),(w,h))

    #Round values in array and cast as 8-bit integer
    imNew = np.array(np.round(imNew),dtype=np.uint8)
    return imNew


# OTHER FUNCTIONS ==============================================================

def stratified_train_test_split(data,target,train_class_size=0.5,
                                test_class_size=None,chop_large_classes=True):
    """Stratified sample a dataset by specified train and test class sizes.

    Parameters
    ----------
    data : array
        Array of data.
    target : array or list
        List of variables by which to group data. Must be same length as data.
    train_class_size : int or float, optional
        Per-class size of outputted training dataset. If float, treated as
        proportion of each class that should go into training set (must be
        between 0 and 1). If int, treated as exact number of individuals per
        class to go into training set.
    test_class_size : int or float, optional
        Per-class size of outputted testing dataset. If float, treated as
        proportion of each class that should go into testing set (must be
        between 0 and 1). If int, treated as exact number of individuals per
        class to go into testing set. If None, all remaining individuals per
        class, not assigned to training set will be assigned to testing set.
    chop_large_classes : bool, default True
        Given a median class size `m`, if True, any class of size >=2m will be
        reduced to size 2m. Serves primarily to reduce the size of the often
        very large `rare_unknown` class.

    Returns
    -------
    train_data : array
        Subset of `data`, where each class is represented by `train_class_size`
        randomly-selected individuals.
    test_data : array
        Subset of `data`, where each class is represented by `test_class_size`
        randomly-selected individuals that are different from `train_data`.
    train_target : array
        Subset of `target` that compliments individuals in `train_data`.
    test_target : array
        Subset of `target` that compliments individuals in `test_data`.

    Note: All classes must have a length >= 'train_class_size +
    test_class_size', otherwise raises ValueError."""

    data = np.array(data).copy()
    target = np.array(target).copy()
    if len(data)!=len(target):
        raise ValueError('`data` and `target` must be same lengths.')

    #df = pd.DataFrame({'targ':target},index=range(len(target)))
    s = pd.Series(target)
    inds = s.groupby(s).indices.values()
    lens = list(map(len,inds))
    if chop_large_classes:
        m = int(np.median(lens))
        lens = [2*m if i>=2*m else i for i in lens]
    n = len(lens)
    if train_class_size is None:
        if type(test_class_size) == int:
            sizes = lens; c = [-test_class_size]*n
        elif type(test_class_size) == float:
            sizes = lens; c=[-int(test_class_size*size) for size in lens]
    elif type(train_class_size) == int:
        if type(test_class_size) == int:
            sizes = [train_class_size+test_class_size]*n
            c = [train_class_size]*n
        elif test_class_size is None:
            sizes = lens; c = [train_class_size ]*n
    elif type(train_class_size) == float:
        if test_class_size is None:
            sizes = lens; c=[int(train_class_size*size) for size in lens]
    inds_new = [np.random.choice(i,size,replace=False) for i,size in zip(inds,sizes)]
    inds_div = [[i[:C],i[C:]] for i,C in zip(inds_new,c)]
    train = np.concatenate(np.array([i[0] for i in inds_div]))
    test = np.concatenate(np.array([i[1] for i in inds_div]))
    return data[train],data[test],target[train],target[test]


def top_k_accuracy_score(y_true, probas_pred, classes, k=5, normalize=True):
    """Computes the top-k classification accuracy score in multilabel
    classification: the number (or proportion) of samples where the true label
    lies among the k highest-predicted labels for that sample.

    This is a helpful measure of accuracy when, rather than presenting a user
    with only the highest-scoring label during classification, the user is to
    be presented with multiple top-ranking labels.

    Parameters
    ----------
    y_true : list or array, shape = [n_samples]
        Ground truth (correct) labels.

    probas_pred : array-like, shape = [n_samples,n_classes]
        Estimated probabilities, e.g. from sklearn_classifier.predict_proba().

    classes : list or array, shape = [n_classes]
        Class labels in order that they are read by classifier (e.g. from
        sklearn_classifier.classes_).

    k : int (default=5)
        Number of top predictions to compare to the true value for each sample.
        Must be 0 < ``k`` <= n_classes.
        Note: if ``k=1``, result will match sklearn.metrics.accuracy_score().

    normalize : bool, optional (default=True)
        If ``False``, returns the number of top-k-correctly classified samples.
        Otherwise, returns the fraction of top-k-correctly classified samples.

    Returns
    -------
    score : float
        If ``normalize == True``, returns the fraction of top-k-correctly
         classified samples (float), else it returns the count of correctly
         classified samples (int).

        The best performance is 1 with ``normalize == True`` and n_samples
        with ``normalize == False``.

    Examples
    --------
    >>>
    """

    # Check shapes of `y_true`, `probas_pred`, and `classes`
    probas_pred = np.asarray(probas_pred)
    if len(y_true) != probas_pred.shape[0]:
        raise ValueError("Number of samples in `y_true` and `probas_pred` don't match.")
    elif probas_pred.shape[1] != len(classes):
        raise ValueError("Length of `classes` must match 2nd dimension of `probas_pred`.")

    # Check `k`
    k = int(k) # Cast `k` to int
    if k>len(classes):
        raise ValueError('`k` must be <= n_classes.')
    elif k==0:
        raise ValueError('`k` must be > zero.')

    # Convert `y_true` to indices representing the labels in `classes`
    class2ind = dict(zip((classes),range(len(classes))))
    y_true_inds = [class2ind[i] for i in y_true]

    # Get indices of k-highest-predicted labels for each sample
    top_k_preds = [np.argsort(i)[::-1][:k] for i in probas_pred]

    # Get list of top-k correct predictions
    in_top_k = [i in j for i,j in zip(y_true_inds,top_k_preds)]

    if normalize:
        return in_top_k.count(True)/float(len(in_top_k))
    else:
        return in_top_k.count(True)


def top_k_precision(y_true, probas_pred, classes, k=5, normalize=True,
                    class_sizes=False):
    """Computes the top-k classification precision for each class in multilabel
    classification: the number (or proportion) of samples where the true label
    lies among the k highest-predicted labels for that sample.

    Note: This function behaves similarly to top_k_accuracy_score, except that
    scores are calculated by class rather than globally.

    Parameters
    ----------
    y_true : list or array, shape = [n_samples]
        Ground truth (correct) labels.

    probas_pred : array-like, shape = [n_samples,n_classes]
        Estimated probabilities, e.g. from sklearn_classifier.predict_proba().

    classes : list or array, shape = [n_classes]
        Class labels in order that they are read by classifier (e.g. from
        sklearn_classifier.classes_).

    k : int (default=5)
        Number of top predictions to compare to the true value for each sample.
        Must be 0 < ``k`` <= n_classes.
        Note: if ``k=1``, result will match sklearn.metrics.accuracy_score().

    normalize : bool, optional (default=True)
        If ``False``, returns the numbers of top-k-correctly classified samples
        by class. Otherwise, returns the fraction of top-k-correctly classified
        samples in each class.

    class_sizes : bool, optional (default=False)
        If ``True``, returns the number of samples in `y_true` for each class,
        equivalent to 'support' in
        sklearn.metrics.precision_recall_fscore_support().

    Returns
    -------
    scores : list
        If ``normalize == True``, returns the fractions of top-k-correctly
        classified samples by class (floats), else it returns the counts of
        correctly classified samples by class (ints). Scores are returned in
        the same order as `classes`.

        The best performance is 1 with ``normalize == True`` and n_samples
        with ``normalize == False``.

    sizes : list
        If ``class_sizes == True``, returns the count of individuals from
        `y_true` that fall into each class in `classes`. Counts are returned in
        the same order as `classes`.

    Examples
    --------
    >>>
    """

    # Check shapes of `y_true`, `probas_pred`, and `classes`
    probas_pred = np.asarray(probas_pred)
    if len(y_true) != probas_pred.shape[0]:
        raise ValueError("Number of samples in `y_true` and `probas_pred` don't match.")
    elif probas_pred.shape[1] != len(classes):
        raise ValueError("Length of `classes` must match 2nd dimension of `probas_pred`.")

    # Check `k`
    k = int(k) # Cast `k` to int
    if k>len(classes):
        raise ValueError('`k` must be <= n_classes.')
    elif k==0:
        raise ValueError('`k` must be > zero.')

    # Convert `y_true` to indices representing the labels in `classes`
    class2ind = dict(zip((classes),range(len(classes))))
    y_true_inds = [class2ind[i] for i in y_true]

    # Get indices of k-highest-predicted labels for each sample
    top_k_preds = [np.argsort(i)[::-1][:k] for i in probas_pred]

    # Get list of top-k correct predictions
    in_top_k = [i in j for i,j in zip(y_true_inds,top_k_preds)]

    # Count top-k correct predictions for & number of samples in each class
    df = pd.DataFrame.from_dict({'class':y_true,'in':in_top_k})
    gb = df.groupby(['class'])
    in_by_class = [gb.get_group(c)['in'].tolist().count(True) for c in classes]
    sizes = [len(gb.get_group(c)) for c in classes]

    if normalize:
        if class_sizes:
            return [b/float(t) for b,t in zip(in_by_class,sizes)],sizes
        else:
            return [b/float(t) for b,t in zip(in_by_class,sizes)]
    else:
        if class_sizes:
            return in_by_class,sizes
        else:
            return in_by_class


def top_k_classification_report(y_true, probas_pred, classes, k=5, digits=2):
    """Builds a text report showing the top-k precision and support, by class,
    similar to sklearn.metrics.classification_report().

    Parameters
    ----------
    y_true : list or array, shape = [n_samples]
        Ground truth (correct) labels.

    probas_pred : array-like, shape = [n_samples,n_classes]
        Estimated probabilities, e.g. from sklearn_classifier.predict_proba().

    classes : list or array, shape = [n_classes]
        Class labels in order that they are read by classifier (e.g. from
        sklearn_classifier.classes_).

    k : int (default=5)
        Number of top predictions to compare to the true value for each sample.
        Must be 0 < ``k`` <= n_classes.
        Note: if ``k=1``, result will match sklearn.metrics.accuracy_score().

    digits : int
        Number of digits for formatting output floating point values

    Returns
    -------
    report : string
        Text summary of the precision and support for each class.

    Examples
    --------
    >>>
    """

    last_line_heading = 'avg / total'

    target_names = ['%s' % l for l in classes]

    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    p, s = top_k_precision(y_true, probas_pred, classes, k, normalize=True,
                           class_sizes=True)

    for i, label in enumerate(classes):
        values = [target_names[i]]
        values += ["{0:0.{1}f}".format(p[i], digits)]
        values += ["{0}".format(s[i])]
        report += fmt % tuple(values)

    report += '\n'

    # compute averages
    values = [last_line_heading]
    values += ["{0:0.{1}f}".format(np.average(p), digits)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)
    return report


def confusion_plot(y_true, y_pred,title=None,label_func=None,ytick_fontsize=8):
    """Plots a confusion matrix to evaluate the accuracy of a classification.
    Uses `sklearn.metrics.confusion_matrix()` to calculate matrix. See
    that function for description of this matrix.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.
    title :
    label_func : function
        Function to be applied individually to label strings
    Returns
    -------
    confusion matrix plot
        Plot using `matplotlib.pyplot`.

    """
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true,y_pred) # calculate conf matrix
    diag = np.diag(cm) # no. samples correctly classified
    counts = np.sum(cm,axis=1) # class counts (row sums)
    cm_norm = (cm.T/counts.astype(float)).T # conf matrix normalized by row

    #Set colormap
    cmap = plt.cm.OrRd#Blues#Reds#OrRd#PuRd#BuGn

    # Set up row labels
    if label_func is not None:
      conditioned_labels = [label_func(l) for l in classes]
    else:
      conditioned_labels = classes
    labels = ['{} ({}/{})'.format(l,d,n) for l,d,n in zip(conditioned_labels,diag,counts)]
    x = range(len(classes))

    fig, ax = plt.subplots(1,1,figsize=(10,5))
    image = ax.imshow(cm_norm,  interpolation='nearest',cmap=cmap)
    fig.colorbar(image,
                 pad=0.01 # amount of padding b/w colorbar & plot (0.05 default)
                 )
    ax.set_title('Confusion matrix')

    # draw subtle gridlines
    l = [i-0.5 for i in x[1:]]
    plt.vlines(l,-0.5,len(classes),
               colors='#C0C0C0',#'w', # silver
               linewidth=0.2)
    plt.hlines(l,-0.5,len(classes),
               colors='#C0C0C0',#'w', # silver
               linewidth=0.2)

    # label axes & ticks
    plt.yticks(x,labels,fontsize=ytick_fontsize)
    plt.xticks([])
    plt.xlabel('Predicted class label')
    plt.ylabel('True class label (Fraction of correctly classified)')

    plt.xlim(-0.5,len(classes)-0.5)
    plt.ylim(len(classes)-0.5,-0.5) # hi to low so image not flipped

    if title is not None:
        plt.title(str(title))
    #plt.show()
