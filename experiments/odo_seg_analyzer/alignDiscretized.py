from skimage.transform import ProjectiveTransform, SimilarityTransform
from skimage.measure import ransac
from skimage.transform import warp
from matplotlib import pyplot as plt
from skimage.feature import ORB, match_descriptors, plot_matches
import numpy as np
import cv2

def alignDiscretized(ref_img_name,ref_img_dir, img_name,img_dir,out_dir,show=False):

    img1name = ref_img_name
    img2name = img_name
    image1 = plt.imread(ref_img_dir + "/" + img1name + "_discrete.png")
    image1 = cv2.cvtColor(image1,cv2.COLOR_RGBA2GRAY)
    image2 = plt.imread(img_dir + "/" + img2name + "_discrete.png")
    image2 = cv2.cvtColor(image2,cv2.COLOR_RGBA2GRAY)

    if(show):
        cv2.imshow('Reference Image', image1)
        cv2.imshow('Image to Align', image2)
        cv2.waitKey(0)

    # Final crop to a common area
    #image1 = image1[50:350,50:350]   # Original image
    #image2 = image2[50:350,50:350]   # Transformed image

    # create orb feature detector
    orb = ORB(n_keypoints=500, fast_threshold=0.05)
    orb.detect_and_extract(image1)
    # keypoints contains location, scale, and rotation of features
    keypoints1 = orb.keypoints
    # descriptors contains visual descriptions of features
    descriptors1 = orb.descriptors
    orb.detect_and_extract(image2)
    keypoints2 = orb.keypoints
    descriptors2 = orb.descriptors
    # match the descriptors and plot matches
    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
    if(show):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_matches(ax, image1, image2, keypoints1, keypoints2, matches12, only_matches=True)
        plt.axis('off')
        plt.show()

    # select keypoints from the source (image to be registered)
    # and target (reference image)
    src = keypoints2[matches12[:, 1]][:, ::-1]
    dst = keypoints1[matches12[:, 0]][:, ::-1]

    # ransac detects outliers in keypoint matches for removal, so we keep only the best matches
    # outputs a similarity transform
    model_robust, inliers = ransac((src, dst), SimilarityTransform,
                                   min_samples=10, residual_threshold=10, max_trials=2000)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_matches(ax, image1, image2, keypoints1, keypoints2, matches12[inliers])
    ax.axis('off')
    plt.show()

    output_shape = image1.shape
    # warp image to reference image using similarity transformation
    image2_ = warp(image2, model_robust.inverse, preserve_range=True,
                   output_shape=output_shape, cval=-1)

    image2_ = np.ma.array(image2_, mask=image2_==-1)

    if(show):
        cv2.imshow('Aligned Image', image2_)
        cv2.waitKey(0)

    cv2.imwrite(out_dir + "/" + img_name + "_aligned.png",image2_)
