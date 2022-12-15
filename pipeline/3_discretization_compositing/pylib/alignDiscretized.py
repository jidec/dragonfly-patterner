from skimage.transform import ProjectiveTransform, SimilarityTransform
from skimage.measure import ransac
from skimage.transform import warp
from matplotlib import pyplot as plt
from skimage.feature import ORB, match_descriptors, plot_matches, SIFT
import numpy as np
import cv2
from showImages import showImages

def alignDiscretized(img,ref_img,show=False):

    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_start = np.copy(img)
    ref_img = cv2.cvtColor(ref_img,cv2.COLOR_RGB2GRAY)

    # create orb feature detector
    orb = SIFT()
    #orb = ORB(n_keypoints=500, fast_threshold=0.05)

    # detect and extract using orb
    orb.detect_and_extract(img)
    # keypoints contains location, scale, and rotation of features
    keypoints_img = orb.keypoints
    # descriptors contains visual descriptions of features
    descriptors_img = orb.descriptors

    orb.detect_and_extract(ref_img)
    keypoints_ref = orb.keypoints
    descriptors_ref = orb.descriptors

    # match the descriptors and plot matches
    matches = match_descriptors(descriptors_ref, descriptors_img, cross_check=True)
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_matches(ax, ref_img, img, keypoints_ref, keypoints_img, matches, only_matches=True)
        plt.axis('off')
        plt.show()

    # select keypoints from the source (image to be registered)
    # and target (reference image)
    src = keypoints_img[matches[:, 1]][:, ::-1]
    dst = keypoints_ref[matches[:, 0]][:, ::-1] #img

    # ransac detects outliers in keypoint matches for removal, so we keep only the best matches
    # outputs a similarity transform
    model_robust, inliers = ransac((src, dst), SimilarityTransform,
                                   min_samples=10, residual_threshold=10, max_trials=2000)
    if show:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_matches(ax, ref_img, img, keypoints_ref, keypoints_img, matches[inliers])
        ax.axis('off')
        plt.show()

    output_shape = ref_img.shape
    # warp image to reference image using similarity transformation
    img = warp(img, model_robust.inverse, preserve_range=False,
                   output_shape=output_shape, cval=0)

    img = np.ma.array(img, mask=img==-1)
    img_start = np.float32(img_start)
    ref_img = np.float32(ref_img)
    img = np.float32(img)
    showImages(show,[img_start,ref_img,img],["Align Image","Reference Image","Aligned"])

    return(img)
