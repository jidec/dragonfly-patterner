import cv2
import numpy as np
import math

def rotateToVertical(img_name,img_dir,out_dir,show=False):
  img = cv2.imread(img_dir + "/" + img_name + "_discrete.png", 0)
  if(show):
    cv2.imshow("Image",img)
    cv2.waitKey(0)

  # convert img to grey
  #img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # set a thresh
  thresh = 3
  # get threshold image
  ret, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

  # find contours
  contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  big_contour = max(contours, key = cv2.contourArea)
  big_ellipse = cv2.fitEllipse(big_contour)

  #cimg = np.zeros_like(img)
  #cv2.drawContours(cimg, contours, 0, color=127, thickness=-1)
  #cv2.imshow("thresh1",cimg)
  #cv2.waitKey(0)

  (xc,yc),(d1,d2),angle = big_ellipse
  print(xc,yc,d1,d1,angle)

  # draw ellipse
  #result = np.zeros_like(img)
  #cv2.ellipse(result, big_ellipse, (0, 50, 0), 3)


  # draw vertical line
  # compute major radius
  rmajor = max(d1,d2)/2
  if angle > 90:
      angle = angle - 90
  else:
      angle = angle + 90
  print(angle)
  xtop = xc + math.cos(math.radians(angle))*rmajor
  ytop = yc + math.sin(math.radians(angle))*rmajor
  xbot = xc + math.cos(math.radians(angle+180))*rmajor
  ybot = yc + math.sin(math.radians(angle+180))*rmajor

  axis_line = (int(xtop),int(ytop)), (int(xbot),int(ybot)), (255, 255, 255), 3
  print(axis_line[1])

  line_img = np.zeros_like(img)
  cv2.line(line_img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (255, 255, 255), 3)
  lines = cv2.HoughLines(line_img, 1, np.pi/180, 175)

  rho, theta = lines[0][0]
  angle = theta*180/np.pi

  (h, w) = img.shape[:2]
  (cX, cY) = (w // 2, h // 2)

  M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])

  nW = int((h * sin) + (w * cos))
  nH = int((h * cos) + (w * sin))

  M[0, 2] += (nW / 2) - cX
  M[1, 2] += (nH / 2) - cY

  img = cv2.warpAffine(img, M, (nW, nH))

  if(show):
    cv2.imshow("Vertically Rotated",img)
    cv2.waitKey(0)

  cv2.imwrite(out_dir + "/" + img_name + "_vert.png",img)

