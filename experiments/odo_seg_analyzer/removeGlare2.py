if remove_glare:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh_img = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.erode(thresh_img, None, iterations=2)
    thresh_img = cv2.dilate(thresh_img, None, iterations=4)
    # perform a connected component analysis on the thresholded image,
    # then initialize a mask to store only the "large" components
    labels = measure.label(thresh_img, background=0)
    glare_mask = np.zeros(thresh_img.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh_img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300:
            glare_mask = cv2.add(glare_mask, labelMask)
    if show:
        cv2.imshow("Glare Mask", glare_mask)
        cv2.imshow("Image", img)
        cv2.waitKey(0)