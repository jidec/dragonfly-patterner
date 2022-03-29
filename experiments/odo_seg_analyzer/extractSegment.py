from PIL import Image
from PIL import ImageOps
import numpy as np
import cv2

# extract segment
def extractSegment(img_name,img_dir,mask_dir,erode=True,erode_kernel_size=4,show=False):
    #img_name = "4155610_198"  # name of image minus extension to perform operations on
    #show = True
    #img_dir = "images"
    #mask_dir = "images"
    #erode=True
    #erode_kernel_size=6
    #show=True

    # open image and convert to array
    img = Image.open(img_dir + "/" + img_name + ".jpg")
    img = np.array(img, dtype=np.uint8)

    if(show):
        cv2.imshow('Image', img)
        cv2.waitKey(0)

    # open mask and convert to array
    mask = Image.open(mask_dir + "/" + img_name + "_mask" + ".jpg")

    mask = ImageOps.invert(mask)  # invert mask so black represents background
    mask = np.array(mask, dtype=np.uint8)
    prev_mask = mask
    
    # erode border of mask
    if(erode):
        kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        mask = cv2.erode(mask, kernel)
        
    if(show):
        cv2.imshow('Mask', prev_mask)
        cv2.imshow('Eroded Mask', mask)
        cv2.waitKey(0)
        cv2.waitKey(0)
        

    # apply mask to get masked img
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # create temp grayscale img, threshold against the black background, use to create a bounding rect
    temp = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(coords)

    # narrow image to bounding rect
    masked_img = masked_img[y:y + h, x:x + w]
    
    bb_mask = mask[y:y + h, x:x + w]

    # add alpha channel, convert remaining black background to alpha
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2BGRA)
    alpha = masked_img[:, :, 3]
    alpha[np.all(masked_img[:, :, 0:3] == (0, 0, 0), 2)] = 0

    # convert final masked image to RGBA, write it, and show it
    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGRA2RGBA)
    #cv2.imwrite(out_dir + "/" + img_name + "_masked.png", masked_img)

    if(show):
        cv2.imshow('Masked', masked_img)
        cv2.waitKey(0)

    # return the segment image as a numpy array and the bounding boxed mask
    # both used in colorDiscretize
    return((masked_img,bb_mask))