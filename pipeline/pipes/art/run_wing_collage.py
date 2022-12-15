import cv2
from makeCollage import makeCollage
import glob
import random
import numpy as np
from makeCollage import overlay_transparent
from PIL import ImageOps
from PIL import Image

pats = [cv2.imread(file,cv2.IMREAD_UNCHANGED) for file in sorted(glob.glob('D:/wing-color/data/patterns/*.png'))]
pats = [p[0:60] for p in pats]
pats = pats[500:1000]
img = pats[0]
img = np.asarray(ImageOps.expand(Image.fromarray(img), border=(0,0,150,0), fill=(0,0,0)))
#cv2.imshow('i',img)
#img = ImageOps.expand(img, border=(10,40,80,120), fill=(0,0,0,0))
#cv2.waitKey(0)
img2 = pats[0]
img2 = np.asarray(ImageOps.expand(Image.fromarray(img2), border=(150,0,0,0), fill=(0,0,0)))
#cv2.imshow('i2',img2)


img3 = cv2.addWeighted(img,1,img2,1,0)
#img3 = overlay_transparent(img,img2,200,0)


#cv2.imshow('i3',img3)
#cv2.waitKey(0)

#random.shuffle(pats)
#makeCollage(pats, n_per_row=45,resize_wh=(50,25),white_bg=False,rotation=45,overlap_wh=(35,3),show=True) #30,12

makeCollage(pats, n_per_row=30,resize_wh=(50,50),white_bg=True,rotation=45,rot_jitter=15,overlap_wh=(20,20),show=True) #30,12
#makeCollage(pats, n_per_row=63,resize_wh=(50,25),white_bg=False,rotation="45",overlap_wh=(33,0),show=True) #30,12

#makeCollage(pats, n_per_row=63,resize_wh=(30,25),white_bg=False,rotation="45",overlap_wh=(12,7),show=True) #30,12

#makeCollage(pats, n_per_row=100,resize_wh=(30,25),white_bg=False,rotation="center",overlap_wh=(16,10),show=True) #30,12