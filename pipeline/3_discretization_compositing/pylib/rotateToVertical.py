import cv2
import numpy as np
import math
from showImages import showImages

# rotate a list of img np arrays to be vertical
def rotateToVertical(imgs,show=False):
  for index, img in enumerate(imgs):

    start_img = np.copy(img)

    # set a thresh
    thresh = 3
    # get threshold image

    ret, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    if thresh_img is not None:
      thresh_img = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2GRAY)

      # find contours
      contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

      # get the biggest contour and fit an ellipse to it
      big_contour = max(contours, key = cv2.contourArea)
      if len(big_contour) < 5:
        return None

      big_ellipse = cv2.fitEllipse(big_contour)

      # get params from ellipse
      (xc,yc),(d1,d2),angle = big_ellipse

      # draw ellipse
      #result = np.zeros_like(img)
      #cv2.ellipse(result, big_ellipse, (0, 50, 0), 3)


      # compute major radius
      rmajor = max(d1,d2)/2
      if angle > 90:
          angle = angle - 90
      else:
          angle = angle + 90
      xtop = xc + math.cos(math.radians(angle))*rmajor
      ytop = yc + math.sin(math.radians(angle))*rmajor
      xbot = xc + math.cos(math.radians(angle+180))*rmajor
      ybot = yc + math.sin(math.radians(angle+180))*rmajor

      # create axis line from values
      axis_line = ((int(xtop),int(ytop)), (int(xbot),int(ybot)), (255, 255, 255), 3)

      # create empty image and draw line
      line_img = np.zeros_like(img,dtype = np.uint8)
      cv2.line(line_img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (255, 255, 255), 3)

      line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)

      # get line in HoughLines format
      lines = cv2.HoughLines(line_img, 1, np.pi/180, 175)

      if lines is not None:
        # get new line parameters
        rho, theta = lines[0][0]
        angle = theta*180/np.pi

        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # get rotation matrix and transform
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # warp image to new rotation
        img = cv2.warpAffine(img, M, (nW, nH))

        # create bounding rect around img
        grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        th = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        coords = cv2.findNonZero(th)
        x, y, w, h = cv2.boundingRect(coords)

        # narrow image to bounding rect
        img = img[y:y + h, x:x + w]

        showImages(show,[start_img,line_img,img],["Discretized Segment","Ellipse Axis Line","Verticalized Segment"])
        #print(img.shape[0])
        bot_half = img[int(img.shape[0]/2):img.shape[0]]
        top_half = img[0:int(img.shape[0]/2)]

        nbot = bot_half[bot_half > 0].size
        ntop = top_half[top_half > 0].size

        if nbot > ntop:
          img = cv2.flip(img, 0)

        imgs[index] = img
  return imgs

