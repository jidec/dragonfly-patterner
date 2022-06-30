import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from os.path import exists
from showImages import showImages

# need function to get image ids without segments/classes, and not annotated yet, infer these and merge into inferences.csv
def inferImageClasses(image_ids, infer_colname, infer_names, model_location, image_size, show):

    image_locs = image_ids
    # turn list of ids into list of locations
    for i in range(0,len(image_locs)):
        image_locs[i] = "../../data/all_images/" + image_locs[i] + ".jpg"

    # load the trained model and set it to eval mode
    model = torch.load(model_location)
    model.eval()

    inferences = []

    for index, img_loc in enumerate(image_locs):
        # read  a sample image
        img = cv2.imread(img_loc)
        # save start dims and resize to input
        img_dims = (img.shape[1],img.shape[0])
        img = cv2.resize(img,(image_size,image_size))

        # this is prolly different
        img = img.transpose(2,0,1).reshape(1,3,image_size,image_size) # cracks are 480 x 320

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with torch.no_grad():
            a = model((torch.from_numpy(img).type(torch.cuda.FloatTensor)/255).to(device))
        max_index = torch.argmax(torch.flatten(a.cpu()), dim=0)
        inferences.append(max_index)
        if show:
            print(infer_names[max_index])
            showImages(show,cv2.imread(img_loc),["Inferred " + infer_names[max_index]])
        print(index)
        # Plot histogram of the prediction to find a suitable threshold. From the histogram a 0.1 looks like a good choice.
        # plt.hist(a['out'].data.cpu().numpy().flatten())
        #plt.show()

        # Plot the input image, ground truth and the predicted output
        #plt.figure(figsize=(10,10));
        #plt.subplot(131);
        #plt.imshow(img[0,...].transpose(1,2,0));
        #plt.title('Image')
        #plt.axis('off');
        #plt.savefig('./out/SegmentationOutput_' + f,bbox_inches='tight')
        #img = a['out'].cpu().detach().numpy()[0][0] > 0.7
        #img = img.astype(np.uint8)  # convert to an unsigned byte
        #img *= 255
        # find size of original image
        #img = cv2.resize(img, img_dims)
        #cv2.imwrite("out/" + imgname + "_segout.png", img)

    inferences = pd.DataFrame({'imageID': image_ids, infer_colname: inferences})

    # replace max weight index with text inference
    inferences.loc[inferences[:,1] == 0, infer_colname] = infer_names[0]
    inferences.loc[inferences[:,1] == 1, infer_colname] = infer_names[1]
    inferences.loc[inferences[:,1] == 2, infer_colname] = infer_names[2]
    inferences.to_csv("../../data/inferences.csv")