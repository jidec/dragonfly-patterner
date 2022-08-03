import numpy as np
import cv2
from skimage import measure
import os
from clahe import equalizeCLAHE

lum_thresh = 200

def create_mask(image):
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
    blurred = cv2.GaussianBlur( gray, (9,9), 0 )
    _,thresh_img = cv2.threshold( blurred, lum_thresh, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.erode( thresh_img, None, iterations=2 )
    thresh_img  = cv2.dilate( thresh_img, None, iterations=4 )
    # perform a connected component analysis on the thresholded image,
    # then initialize a mask to store only the "large" components
    labels = measure.label(thresh_img) #, neighbors=8, background=0 )
    mask = np.zeros( thresh_img.shape, dtype="uint8" )
    # loop over the unique components
    for label in np.unique( labels ):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros( thresh_img.shape, dtype="uint8" )
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero( labelMask )
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add( mask, labelMask )
    return mask

imgs = os.listdir("../../data/all_images")
imgs = imgs[1100:]
imgs = ["INATRANDOM-102083730.jpg","INAT-54529717-1.jpg","INAT-54622783-1.jpg","INATRANDOM-111036639.jpg"]
for img_name in imgs:

    img = equalizeCLAHE('../../data/all_images/' + img_name)
    #img = cv2.imread('../../data/all_images/' + img_name)

    # resize image
    img = cv2.resize(img, (1000,1000), interpolation=cv2.INTER_AREA)

    mask = create_mask(img)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imshow('mask',mask)
    cv2.waitKey(0)

    dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
    cv2.imshow('inpaint',dst)


    cv2.waitKey(0)
    cv2.destroyAllWindows()