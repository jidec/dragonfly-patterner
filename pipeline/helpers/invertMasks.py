import os
import cv2

def invertMasks(targetdir,invert_if_more_white=False):
    filenames = os.listdir(targetdir)
    for f in filenames:
        img = cv2.imread(targetdir + "/" + f,cv2.IMREAD_GRAYSCALE)
        white_count = cv2.countNonZero(img)

        # invert if not using more white criterion or using criterion and white count is greater than half
        # only works for segmenting tasks when the segment will always be smaller than the background by pixel number
        if not invert_if_more_white or (invert_if_more_white and white_count / (img.shape[0] * img.shape[1]) > 0.5):
            img = cv2.bitwise_not(img)
            cv2.imwrite(targetdir + "/" + f,img)