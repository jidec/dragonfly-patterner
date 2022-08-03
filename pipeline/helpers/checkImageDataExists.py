import os
import cv2
from getFilterImageIDs import getFilterImageIDs

def checkValidImageData(direct_data_dir):
    data_names = os.listdir(direct_data_dir)
    for name in data_names:
        img = cv2.imread(direct_data_dir + "/" + name)
        if img is None:
            print(name + " is None")

checkValidImageData("../../data/masks")

def checkValidImageDataFromIDs(image_ids, direct_data_dir, suffix):
    data_names = [i + suffix for i in image_ids]
    for name in data_names:
        img = cv2.imread(direct_data_dir + "/" + name)
        if img is None:
            print(name + " is None")

ids = getFilterImageIDs(records_fields=["genus"],records_values=["Dythemis"])
ids = ids[2:9]
checkValidImageDataFromIDs("../../data/masks")

def imageDataNameToImageID(name):
    return name.split("_")[0]
