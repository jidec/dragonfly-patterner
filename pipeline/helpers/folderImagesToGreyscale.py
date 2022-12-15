import cv2
import pandas as pd
import os
import glob

def folderImagesToGreyscale(direct_dir):
    files = glob.glob(direct_dir + '/*.jpg')
    for f in files:
        img = cv2.imread(f,cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(f, img)

#folderImagesToGreyscale("E:/dragonfly-patterner/data/other/training_dirs/segmenter/train/image")

