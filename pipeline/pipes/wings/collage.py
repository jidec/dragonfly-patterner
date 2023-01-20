import cv2
from makeCollage import makeCollage
import glob

#segs = [cv2.imread(file) for file in sorted(glob.glob('D:/wing-color/data/patterns/*.png'))]
#segs = segs[1:100]
#makeCollage(segs, n_per_row=20,resize_wh=(35,250),show=True)

#pats = [cv2.imread(file) for file in sorted(glob.glob('E:/dragonfly-patterner/data/patterns/*.png'))]
#makeCollage(pats, n_per_row=20,resize_wh=(35,250),show=True)

pats = [cv2.imread(file) for file in sorted(glob.glob('D:/wing-color/data/patterns/grouped2/*.png'))]
segs = [cv2.imread(file) for file in sorted(glob.glob('D:/wing-color/data/segments/*.png'))]
pats = pats[0:10]
segs = segs[0:10]

pairs = []
for i in range(len(pats)):
    pair = makeCollage([segs[i],pats[i]],resize_wh=(300,75),n_per_row=1,show=True)
    pairs.append(pair)

makeCollage(pairs, n_per_row=2,resize_wh=(300,300),show=True)
