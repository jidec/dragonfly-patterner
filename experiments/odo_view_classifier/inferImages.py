import cv2
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import numpy as np

def inferImages(image_dir, model_location,image_size,show):

    # get list of images in location
    file_list = os.listdir(image_dir)

    # load the trained model and set it to eval mode
    model = torch.load(model_location)
    model.eval()

    for f in file_list:
        imgname = f.replace(".jpg", "", 1)

        # read  a sample image
        img = cv2.imread(image_dir + '/' + imgname + '.jpg')

        # save start dims and resize to input
        img_dims = (img.shape[1],img.shape[0])
        img = cv2.resize(img,(image_size,image_size))

        if(show):
            cv2.imshow("img", img)
            cv2.waitKey(0)

        # this is prolly different
        img = img.transpose(2,0,1).reshape(1,3,image_size,image_size) # cracks are 480 x 320

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        with torch.no_grad():
            a = model((torch.from_numpy(img).type(torch.cuda.FloatTensor)/255).to(device))
        print(a)
        print(torch.max(a))
        # return the label

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