import cv2
import numpy as np
from alignDiscretized import alignDiscretized
from blendAligned import blendAligned
import random

def procrustesComposite(image_ids, procrustes_margin=1000,show=False, img_dir="../../data/patterns/individuals"):
    imgs = []
    for id in image_ids:
        # open image and convert to array
        img = cv2.imread(img_dir + "/" + id + "_pattern.jpg", cv2.IMREAD_COLOR)
        imgs.append(img)

    procrustes_diff = 100000000
    # select an arbitrary shape as the mean reference shape
    ref = random.choice(imgs)
    while procrustes_diff > procrustes_margin:
        # align all other shapes to the mean reference shape
        all_aligned = []
        for img in imgs:
            aligned = alignDiscretized(img,ref,show)
            all_aligned.append(aligned)
        blended = blendAligned(all_aligned)
        procrustes_diff = np.abs(blended - ref)
        if procrustes_diff > procrustes_margin:
            ref = alignDiscretized(blended,ref,show)

    composite = ref
    return(composite)

