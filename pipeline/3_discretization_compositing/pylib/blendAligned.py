import cv2
import os

def blendAligned(imgs,show=False):
    # loop through image list and blend all
    for i, img in enumerate(imgs):
        if i == 0:
            first_img = img
            continue
        else:
            second_img = img
            second_weight = 1/(i+1)
            first_weight = 1 - second_weight
            first_img = cv2.addWeighted(first_img, first_weight, second_img, second_weight, 0)

    blended = first_img
    return(blended)