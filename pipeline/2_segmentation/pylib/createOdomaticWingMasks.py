import pandas as pd
import numpy as np
import cv2
from odomatic_helpers import _masker_v4_3, _masker_v4_2, _masker_v1
from showImages import showImages
from skimage.filters import roberts
import imutils
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

def createOdomaticWingMasks(image_ids,show=False,proj_dir="../.."):

    for id in image_ids:
        img = cv2.imread(proj_dir + "/data/all_images/" + id + ".tif", cv2.IMREAD_COLOR)
        if img is None:
            continue
        crop_height = int(img.shape[0] * 0.5)
        img_top = img[0:crop_height]
        img_top = cv2.copyMakeBorder(img_top,top=20,bottom=20,left=20,right=20,borderType=cv2.BORDER_CONSTANT,value=[255, 255, 255])

        img_bot = img[crop_height:img.shape[0]]

        #mask_top = meanShiftOtsuMask(img_top)
        #mask_bot = meanShiftOtsuMask(img_bot)
        mask_top = adaptiveMorphMask(img_top)
        mask_bot = adaptiveMorphMask(img_bot)

        seg_top = cv2.bitwise_and(img_top, img_top,mask_top)
        seg_bot = cv2.bitwise_and(img_bot, img_bot, mask_bot)
        seg_bot = addAlpha(seg_bot)
        seg_top = addAlpha(seg_top)
        showImages(show,images=[img_top,img_bot,mask_top,mask_bot,seg_top,seg_bot])
        cv2.imwrite(proj_dir + "/data/segments/" + id + "_fore_segment.png", seg_bot)
        cv2.imwrite(proj_dir + "/data/segments/" + id + "_hind_segment.png", seg_top)

def meanShiftOtsuMask(img):
    image = img
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    cv2.imshow("Input", image)
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 240, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    mask = cv2.bitwise_not(mask)
    return(mask)

def adaptiveMorphMask(img):
    # Read image
    hh, ww = img.shape[:2]
    # threshold on black
    # Define lower and uppper limits of what we call "white-ish"
    # lower = np.array([white_lower, white_lower, white_lower])
    # upper = np.array([255, 255, 255])

    # Create mask to only select black
    # thresh = cv2.inRange(img, lower, upper)

    # invert mask so shapes are white on black background
    # thresh_inv = 255 - thresh
    start_img = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # (T, thresh) = cv2.threshold(img, 0, 255,
    #                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                   cv2.THRESH_BINARY, 11, 2)
    # ret, thresh = cv2.threshold(img, 127, 255, 0)

    # get the largest contour
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    # big_contour = max(contours, key=cv2.contourArea)
    contours = sorted(contours, key=cv2.contourArea)
    # big_contour = contours[0]
    contours = contours[-2]

    # import cv2
    # all_points = []
    # for ctr in contours:
    #    all_points += [pt[0] for pt in ctr]
    # contour = np.array(all_points).reshape((-1, 1, 2)).astype(np.int32)

    contour = np.vstack(contours)
    # contour = cv2.convexHull(contour)  # done.

    # draw white contour on black background as mask
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=3)
    return(mask)

def addAlpha(img):
    # Make a True/False mask of pixels whose BGR values sum to more than zero
    alpha = np.sum(img, axis=-1) < (255 * 3)
    # Convert True/False to 0/255 and change type to "uint8" to match "na"
    alpha = np.uint8(alpha * 255)
    # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
    img = np.dstack((img, alpha))
    return(img)
createOdomaticWingMasks(['MLM-000001'],proj_dir="D:/wing-color")