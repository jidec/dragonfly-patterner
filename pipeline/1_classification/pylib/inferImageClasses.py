import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from os.path import exists
from showImages import showImages

def inferImageClasses(image_ids, infer_colname, infer_names,model_name, image_size=256,
                      show=False, print_steps=True,print_details=False, proj_dir="../.."):
    """
        Load and train a classification model

        :param List image_ids: the name of the training dir within data/other/training_dirs
        :param str infer_colname: the name of the new column to add to inferences
        :param List infer_names: an ordered list of class strings, used to convert outputs 0,1,2,3.. etc to strings
        :param int num_workers: the number of workers
        :param str model_name: the model in data/ml_models to use
        :param Dictionary data_transforms: instructions for transforming the data, see default example
        :param pretrained_model: the pretrained model, either models.resnet18(pretrained=True) or models.inception_v3(pretrained=True)
    """

    image_locs = image_ids.copy()
    # turn list of ids into list of locations
    for i in range(0,len(image_locs)):
        image_locs[i] = "../../data/all_images/" + image_locs[i] + ".jpg"

    # load the trained model and set it to eval mode
    model = torch.load(proj_dir + "/data/ml_models/" + model_name + ".pt")
    model.eval()
    if print_steps: print("Loaded trained model")

    if print_steps: print("Starting loop over images")
    inferences = []
    weights0 = []
    weights1 = []
    weights2 = []
    weights3 = []
    for index, img_loc in enumerate(image_locs):
        # read  a sample image
        img = cv2.imread(img_loc)
        # save start dims and resize to input
        img_dims = (img.shape[1],img.shape[0])
        img = cv2.resize(img,(image_size,image_size))

        # this is prolly different
        img = img.transpose(2,0,1).reshape(1,3,image_size,image_size) # cracks are 480 x 320
        if print_details: print("Read, resized, and reshaped image")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with torch.no_grad():
            a = model((torch.from_numpy(img).type(torch.cuda.FloatTensor)/255).to(device))

        max_index = torch.argmax(torch.flatten(a.cpu()), dim=0)
        weights_list = torch.flatten(a.cpu()).tolist()
        if print_details: print("Retrieved inference")

        inferences.append(max_index.item())
        weights0.append(weights_list[0])
        weights1.append(weights_list[1])
        weights2.append(weights_list[2])
        weights3.append(weights_list[3])

        showImages(show,cv2.imread(img_loc),["Inferred " + infer_names[max_index]])

        if index > 0 and index % 1000 == 0:
            if print_steps: print(index)
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


    inferences = pd.DataFrame({'imageID': image_ids, infer_colname: inferences,
                               'weight' + infer_names[0]: weights0,
                               'weight' + infer_names[1]: weights1,
                               'weight' + infer_names[2]: weights2,
                               'weight' + infer_names[3]: weights3})

    # fill in inference col (0..4) with text name
    for index, name in enumerate(infer_names):
        inferences.loc[inferences[infer_colname] == index, infer_colname] = name

    if exists(proj_dir + "/data/inferences.csv"):
        current_infers = pd.read_csv(proj_dir + "/data/inferences.csv",header=None)
        inferences = pd.concat([current_infers,inferences])

    inferences.to_csv(proj_dir + "/data/inferences.csv")
    if print_steps: print("Saved all inferences")