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

def createOdomaticWingMasks(image_ids,show=True,proj_dir="../.."):
    #mc_df = pd.read_csv(proj_dir + "/data/other/mask_contours.csv")\

    for id in image_ids:
        img = cv2.imread(proj_dir + "/data/all_images/" + id + ".tif", cv2.IMREAD_COLOR)
        crop_height = int(img.shape[0] * 0.5)
        img_top = img[0:crop_height]
        img_top = cv2.copyMakeBorder(img_top,top=20,bottom=20,left=20,right=20,borderType=cv2.BORDER_CONSTANT,value=[255, 255, 255])

        img_bot = img[crop_height:img.shape[0]]
        cv2.imshow('output', img_top)
        cv2.waitKey(0)
        #cv2.imshow('output', img_bot)
        #cv2.waitKey(0)

        #seg_top = findWingContour(img_top)
        #seg_bot = findWingContour(img_bot)

        #showImages(show=show,images=[img_top,img_bot,seg_top,seg_bot])
        #img_top = img[0 + h_offset:int(crop_height / 2) - h_offset, 0 + w_offset:crop_width - w_offset]
        #img_bot = img[int(crop_height / 2) + h_offset:crop_height - h_offset, 0 + w_offset:crop_width - w_offset]

        #mask_top = watershedSegmenter(img_top)

        #mask_top = _masker_v4_2(img_top,1,convex_hull=False,bgr=True)
        #mask_top = _masker_v4_3(img_top,1, convex_hull=False, bgr=True)

        mask_top = otsu(img_top)
        cv2.imshow('output', mask_top)
        cv2.waitKey(0)
        seg_top = cv2.bitwise_and(img_top, img_top,mask_top)
        cv2.imshow('output2', seg_top)
        cv2.waitKey(0)
        #seg_bot = cv2.bitwise_and(img_bot, img, mask=mask_bot.astype(np.uint8) * 255)

        #showImages([img_top,img_bot,mask_top,mask_bot,seg_top,seg_bot])
        #ss = mc_df.loc[mc_df['uniq_id'] == id]
        #fore = ss.loc[ss['wing'] == 'forewing']
        #hind = ss.loc[ss['wing'] == 'hindwing']
        #fore = fore[['x','y']]
        #hind = hind[['x', 'y']]
        #fore = [np.array(fore,dtype=np.int32)]
        #print(fore)
        #drawing = np.zeros([100, 100], np.uint8)
        #for cnt in fore:
        #    cv2.drawContours(drawing, [cnt], 0, (255, 255, 255), 2)
def otsu(img):
    image = img
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    cv2.imshow("Input", image)
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 240, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return(thresh)
def findWingContour(img,write=True,show=True):
    # Read image
    hh, ww = img.shape[:2]
    # threshold on black
    # Define lower and uppper limits of what we call "white-ish"
    #lower = np.array([white_lower, white_lower, white_lower])
    #upper = np.array([255, 255, 255])

    # Create mask to only select black
    #thresh = cv2.inRange(img, lower, upper)

    # invert mask so shapes are white on black background
    #thresh_inv = 255 - thresh
    start_img = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #(T, thresh) = cv2.threshold(img, 0, 255,
    #                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                               cv2.THRESH_BINARY, 11, 2)
    #ret, thresh = cv2.threshold(img, 127, 255, 0)

    # get the largest contour
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #contours = contours[0] if len(contours) == 2 else contours[1]
    #big_contour = max(contours, key=cv2.contourArea)
    contours = sorted(contours, key=cv2.contourArea)
    #big_contour = contours[0]
    contours = contours[-2]

    #import cv2
    #all_points = []
    #for ctr in contours:
    #    all_points += [pt[0] for pt in ctr]
    #contour = np.array(all_points).reshape((-1, 1, 2)).astype(np.int32)

    contour = np.vstack(contours)
    #contour = cv2.convexHull(contour)  # done.

    # draw white contour on black background as mask
    mask = np.zeros((hh,ww), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)

    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25,25)),iterations=5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=2)

    if (show):
        cv2.imshow("", thresh)
        cv2.waitKey(0)

        cv2.imshow("",mask)
        cv2.waitKey(0)

    # apply mask to image
    image_masked = cv2.bitwise_and(start_img, start_img, mask=mask)

    # apply inverse mask to background
    #bckgnd_masked = cv2.bitwise_and(bckgnd, bckgnd, mask=mask_inv)

    # add together
    #result = cv2.add(image_masked, bckgnd_masked)

    # save results
    #cv2.imwrite('shapes_inverted_mask.jpg', mask_inv)
    cv2.imwrite('shapes_masked.jpg', image_masked)
    #cv2.imwrite('shapes_bckgrnd_masked.jpg', bckgnd_masked )
    #cv2.imwrite('shapes_result.jpg', result)

    #cv2.imshow('mask', mask)
    if (show):
        cv2.imshow('image_masked', image_masked)
        #cv2.imshow('bckgrnd_masked', bckgnd_masked)
        #cv2.imshow('result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return(image_masked)

def watershedSegmenter(img,show=True):
    from matplotlib import pyplot as plt
    #img = cv2.imread('coins.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    #img[markers == -1] = [255, 0, 0]

    img2 = img.copy()
    markers1 = markers.astype(np.uint8)
    ret, m2 = cv2.threshold(markers1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('m2', m2)
    cv2.waitKey(0)
    _, contours, hierarchy = cv2.findContours(m2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        #    img2 = img.copy()
        #    cv2.waitKey(0)
        cv2.drawContours(img2, c, -1, (0, 255, 0), 2)

    # cv2.imshow('markers1', markers1)
    cv2.imshow('contours', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return(img)

def watershedSegmenter2(img):
    # construct the argument parse and parse the arguments
    # load the image and perform pyramid mean shift filtering
    # to aid the thresholding step
    image = img
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    cv2.imshow("Input", image)
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
                              labels=thresh)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

createOdomaticWingMasks(['MLM-000001'],proj_dir="D:/wing-color")