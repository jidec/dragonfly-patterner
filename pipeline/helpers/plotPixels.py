import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def plotPixels(img,centroids=None):
    #convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pixels = []
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            pixel = img[i][j]
            if not (pixel[1] == 255):
                pixel = [p / 255 for p in pixel]
                pixels.append(pixel)

    #get rgb values from image to 1D array
    r, g, b = cv2.split(img)
    r = r.flatten()
    g = g.flatten()
    b = b.flatten()

    #plotting
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(r, g, b,c=pixels,alpha=0.5)
    #print(centroids)
    if centroids is not None:
        ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],s=600,c='black',marker='x')
    plt.show()