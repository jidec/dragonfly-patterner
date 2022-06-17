import os
import cv2

def invertMasks(targetdir):
    filenames = os.listdir(targetdir)
    for f in filenames:
        img = cv2.imread(targetdir + "/" + f)
        img = cv2.bitwise_not(img)
        cv2.imwrite(targetdir + "/" + f,img)
