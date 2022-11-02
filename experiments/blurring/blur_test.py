from skimage import data
from skimage.morphology import disk
from skimage.filters import median, wiener
from showImages import showImages
import cv2
import glob
img = data.camera()
imgs = [cv2.imread(file) for file in glob.glob('../patternize_recolorize/segments/*.png')] #cv2.IMREAD_GRAYSCALE #/example_clade_2
for img in imgs:
    #img = cv2.imread("../patternize_recolorize/segments/INATRANDOM-210309_segment.png",cv2.IMREAD_GRAYSCALE)
    #img = data.camera()
    #med = median(img, disk(3))
    med = cv2.medianBlur(img,5)
    #med = img
    #wien = wiener(img)
    print(img.shape)
    sc = img.shape[0] / 22
    ss = sc
    d = int(img.shape[0] / 25)
    print(d)
    print(sc)
    print(ss)
    bil_adj_d = cv2.bilateralFilter(img, d=d, sigmaColor=75, sigmaSpace=75)
    bil_adj = cv2.bilateralFilter(img, d=d, sigmaColor=sc, sigmaSpace=ss)
    bil = cv2.bilateralFilter(img, d=15, sigmaColor= 75, sigmaSpace= 75)

    bil_space = cv2.bilateralFilter(img, d=d, sigmaColor=25, sigmaSpace=75)
    bil_col = cv2.bilateralFilter(img, d=d, sigmaColor=90, sigmaSpace=25)
    #guid = cv2.ximgproc.GuidedFilter.filter(img)
    #ani = cv2.ximgproc.anisotropicDiffusion(img,alpha=1,K=1,niters=10)
    #print(guid)
    #ss = cv2.ximgproc.edgePreservingFilter(img,d=3)
    #showImages(True,[img,med,bil,bil_adj_d,bil_adj],['img','med','bil','bil_adj_d','bil_adj'])
    #showImages(True, [bil, bil_adj_d, bil_adj], ['bil', 'bil_adj_d', 'bil_adj'])
    showImages(True, [bil, bil_space, bil_col], ['bil', 'bil_space', 'bil_col'])