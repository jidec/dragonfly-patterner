import cv2
import numpy as np

img_locs = ["D:\wing-color\data\segments\MLM-000047_fore_segment.png", # standard
            "D:\wing-color\data\segments\MLM-000054_fore_segment.png", # with yellow
            "D:\wing-color\data\segments\MLM-000084_hind_segment.png", # some brown
            "D:\wing-color\data\segments\WRK-000030_fore_segment.png",
            "D:\wing-color\data\segments\WRK-WS-02331_fore_segment.png"] # some brown and black

for loc in img_locs:
    # load the image as greyscale
    img = cv2.imread(loc, 0)

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, 0]], np.float32)
    sharp = cv2.filter2D(img, -1, kernel)

    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,blockSize=15,C=2)

    thresh2 = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                   cv2.THRESH_BINARY, blockSize=15, C=2)

    # Find contours of the binary image
    contours, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour (the black shape)
    largest_contour = max(contours, key=cv2.contourArea)
    # Create a mask of the largest contour
    mask = np.zeros_like(thresh2)
    cv2.drawContours(mask, [largest_contour], 0, color=255, thickness=-1)
    # Apply the mask to the original image

    # Display the result
    cv2.imshow('Original Image', img)
    cv2.imshow('Binarized Image', thresh)
    cv2.imshow('Binarized Image2', thresh2)
    cv2.imshow('Binarized Image3', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()