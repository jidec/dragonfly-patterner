import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from show_anns import show_anns, show_each_anns

ids = ["INAT-12485-1.jpg","INAT-27596-1.jpg","INAT-2580-1.jpg", "INAT-3489-1.jpg"]
for id in ids:
    image = cv2.imread("../../data/all_images/" + id)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
    cv2.imshow("0",image)
    cv2.waitKey(0)
    sam_checkpoint = "../../data/ml_models/sam.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam,points_per_side=24,pred_iou_thresh=0.93,
                                               stability_score_thresh=0.93,min_mask_region_area=400)
    mask_generator = SamAutomaticMaskGenerator()
    masks = mask_generator.generate(image)

    print(len(masks))
    print(masks[0].keys())

    show_each_anns(masks,image)

    #plt.figure(figsize=(20,20))
    #plt.imshow(image)
    #show_anns(masks)
    #plt.axis('off')
    #plt.show()