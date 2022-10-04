import cv2
import os

def makeTrainingImagesGreyscale(training_dir,proj_dir="../.."):
    target = proj_dir + "/data/other/training_dirs/" + training_dir
    for path, dir, files in os.walk(target):
        for file in files:
            loc = os.path.join(path, file)
            img = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(loc, img)