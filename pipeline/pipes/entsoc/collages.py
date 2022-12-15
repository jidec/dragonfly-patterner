import cv2
from makeCollage import makeCollage
import glob

segs = [cv2.imread(file) for file in sorted(glob.glob('E:/dragonfly-patterner/data/segments/*.png'))]
makeCollage(segs, n_per_row=20,resize_wh=(35,250),show=True)

pats = [cv2.imread(file) for file in sorted(glob.glob('E:/dragonfly-patterner/data/patterns/*.png'))]
makeCollage(pats, n_per_row=20,resize_wh=(35,250),show=True)

pats = [cv2.imread(file) for file in sorted(glob.glob('E:/dragonfly-patterner/data/patterns/gomph_grouped_none/*.png'))]
segs = [cv2.imread(file) for file in sorted(glob.glob('E:/dragonfly-patterner/data/segments/*.png'))]

pairs = []
for i in range(len(pats)):
    pair = makeCollage([segs[i],pats[i]],n_per_row=2,resize_wh=(70,500),show=False)
    pairs.append(pair)

makeCollage(pairs, n_per_row=10,resize_wh=(140,400),show=True)
