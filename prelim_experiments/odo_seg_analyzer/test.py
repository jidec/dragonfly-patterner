import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt

with Image.open("segs/4155610_198_masked.jpg") as img:
    img = np.array(img, dtype=np.uint8)
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    img = res2

    # Coinvert from BGR to BGRA
    img = cv.cvtColor(img, cv.COLOR_LAB2RGB)
    img = cv.cvtColor(img, cv.COLOR_RGB2RGBA)

    # Slice of alpha channel
    alpha = img[:, :, 3]

    # Use logical indexing to set alpha channel to 0 where RGB=0
    alpha[np.all(img[:, :, 0:3] == (0, 0, 0), 2)] = 0

    #img = cv.cvtColor(img, cv.COLOR_LAB2RGB)

    cv.imshow('res2',img)
    cv.waitKey(0)
    plt.hist(res2.ravel(), 5, [0, 3]);
    plt.show()
    cv.imwrite("segs/4155610_198_masked_clust2.png", img)
    cv.waitKey(0)
    cv.destroyAllWindows()