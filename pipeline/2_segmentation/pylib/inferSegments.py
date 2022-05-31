import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from os.path import exists
from showImages import showImages

# need function to get image ids without segments/classes, and not annotated yet, infer these and merge into inferences.csv
def inferSegments(image_ids, model_location,image_size, activation_threshold=0.7, export_dir="../../data/masks", show=False, proj_dir="../.."):

    image_locs = image_ids
    # turn list of ids into list of locations
    for i in range(0,len(image_locs)):
        image_locs[i] = proj_dir + "/data/all_images/" + image_locs[i] + ".jpg"

    # for testing
    #image_locs = os.listdir("../../data/test_images")
    #for i in range(0,len(image_locs)):
    #    image_locs[i] = "../../data/test_images/" + image_locs[i]

    # load the trained model and set it to eval mode
    model = torch.load(model_location)
    model.eval()

    inferences = []

    for index, img_loc in enumerate(image_locs):
        # read  a sample image
        img = cv2.imread(img_loc)

        # save start dims and resize to input
        img_dims = (img.shape[1], img.shape[0])
        start_img = np.copy(img)
        img = cv2.resize(img, (image_size, image_size)) #344

        img = img.transpose(2, 0, 1).reshape(1, 3, image_size, image_size)  #344 # cracks are 480 x 320

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with torch.no_grad():
            a = model((torch.from_numpy(img).type(torch.cuda.FloatTensor) / 255).to(device))

        mask = a['out'].cpu().detach().numpy()[0][0] > activation_threshold
        mask = mask.astype(np.uint8)  # convert to an unsigned byte
        mask *= 255
        mask = cv2.resize(mask,img_dims)
        showImages(show,[start_img,mask],["Image","Inferred Mask"])
        cv2.imwrite(export_dir + "/" + str(index) + "_mask.jpg",mask)