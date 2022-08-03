import cv2
import numpy as np
import time
import os

# modified from user Bharath Kumar on stackoverflow
# https://stackoverflow.com/questions/43470569/remove-glare-from-photo-opencv

def equalizeCLAHE(img_location):
    clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))

#imgs = os.listdir("../../data/all_images")
#imgs = imgs[1100:]
#for img_name in imgs:
    img = cv2.imread(img_location)

    t1 = time.time()
    img = img.copy()

    ## crop if required
    #FACE
    x,y,h,w = 550,250,400,300
    # img = img[y:y+h, x:x+w]

    #NORMAL
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayimg = gray


    GLARE_MIN = np.array([0, 0, 50],np.uint8)
    GLARE_MAX = np.array([0, 0, 225],np.uint8)

    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #HSV
    frame_threshed = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)


    #INPAINT
    mask1 = cv2.threshold(grayimg , 220, 255, cv2.THRESH_BINARY)[1]
    result1 = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA)



    #CLAHE
    claheCorrecttedFrame = clahefilter.apply(grayimg)

    #COLOR
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


    #INPAINT + HSV
    result = cv2.inpaint(img, frame_threshed, 0.1, cv2.INPAINT_TELEA)


    #INPAINT + CLAHE
    grayimg1 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    mask2 = cv2.threshold(grayimg1 , 220, 255, cv2.THRESH_BINARY)[1]
    result2 = cv2.inpaint(img, mask2, 0.1, cv2.INPAINT_TELEA)



    #HSV+ INPAINT + CLAHE
    lab1 = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    lab_planes1 = list(cv2.split(lab1))
    clahe1 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes1[0] = clahe1.apply(lab_planes1[0])
    lab1 = cv2.merge(lab_planes1)
    clahe_bgr1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)




    # fps = 1./(time.time()-t1)
    # cv2.putText(clahe_bgr1    , "FPS: {:.2f}".format(fps), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255))

    # display it
    cv2.imshow("IMAGE", img)
    #cv2.imshow("GRAY", gray)
    #cv2.imshow("HSV", frame_threshed)
    #cv2.imshow("CLAHE", clahe_bgr)
    #cv2.imshow("LAB", lab)
    #cv2.imshow("HSV + INPAINT", result)
    #cv2.imshow("INPAINT", result1)
    #cv2.imshow("CLAHE + INPAINT", result2)
    cv2.imshow("HSV + INPAINT + CLAHE   ", clahe_bgr1)
    cv2.waitKey()
    return(clahe_bgr1)

    #cv2.destroyAllWindows()