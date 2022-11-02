import cv2
import numpy as np
from showImages import showImages
import glob

def makeCollage(imgs,n_per_row,resize_wh,show):
    imgs = [cv2.resize(im,resize_wh) for im in imgs]
    print(len(imgs))
    collage = None

    i = 0
    row_imgs = []
    while(i < len(imgs)):
        print(i)
        # add new img to row
        row_imgs.append(imgs[i])
        # if length of the new row is a multiple of n per row
        if len(row_imgs) % n_per_row == 0:
            print("Adding and resetting row")
            # make a collage of the new row
            row_collage = cv2.hconcat(row_imgs)

            # append to the old collage if it exists
            if collage is None:
                collage = row_collage

            else:
                collage = cv2.vconcat([collage,row_collage])
            # start a new row
            row_imgs = []
        i+=1
    if show:
        cv2.imshow('Collage', collage)
        cv2.waitKey(0)
    return collage

segs = [cv2.imread(file) for file in sorted(glob.glob('E:/dragonfly-patterner/data/segments/*.png'))]
makeCollage(segs, n_per_row=20,resize_wh=(35,250),show=True)

pats = [cv2.imread(file) for file in sorted(glob.glob('E:/dragonfly-patterner/data/patterns/*.png'))]
makeCollage(pats, n_per_row=20,resize_wh=(35,250),show=True)

#images = [cv2.imread(file) for file in glob.glob('E:/dragonfly-patterner/experiments/patternize_recolorize/segments/*.png')]
pats = [cv2.imread(file) for file in sorted(glob.glob('E:/dragonfly-patterner/data/patterns/gomph_grouped_none/*.png'))]
segs = [cv2.imread(file) for file in sorted(glob.glob('E:/dragonfly-patterner/data/segments/*.png'))]
#segs = segs[0:49]

pairs = []
for i in range(len(pats)):
    pair = makeCollage([segs[i],pats[i]],n_per_row=2,resize_wh=(70,500),show=False)
    pairs.append(pair)

makeCollage(pairs, n_per_row=10,resize_wh=(140,400),show=True)

