import cv2
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import numpy as np

# TODO- batch inference

img_size = 344

file_list = os.listdir('data/Test/Image')
# Load the trained model
model = torch.load('segmodel_weights.pt')
# Set the model to evaluate mode
model.eval()

for f in file_list:
    imgname = f.replace(".jpg", "", 1)
    #imgname = "14246470_755"
    # Read  a sample image and mask from the data-set
    img = cv2.imread('data/Test/Image/' + imgname + '.jpg')
    img_dims = (img.shape[1],img.shape[0])
    img = cv2.resize(img,(344,344))
    #cv2.imshow('image',img)
    #cv2.waitKey(0)

    img = img.transpose(2,0,1).reshape(1,3,344,344) # cracks are 480 x 320
    mask = cv2.imread('data/Test/Mask/' + imgname + '_mask.jpg')
    mask = cv2.resize(mask,(344,344))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #img.to(device)
    model.to(device)

    with torch.no_grad():
        a = model((torch.from_numpy(img).type(torch.cuda.FloatTensor)/255).to(device))

    # Plot histogram of the prediction to find a suitable threshold. From the histogram a 0.1 looks like a good choice.
    plt.hist(a['out'].data.cpu().numpy().flatten())
    #plt.show()

    # Plot the input image, ground truth and the predicted output
    plt.figure(figsize=(10,10));
    plt.subplot(131);
    plt.imshow(img[0,...].transpose(1,2,0));
    plt.title('Image')
    plt.axis('off');
    plt.subplot(132);
    plt.imshow(mask);
    plt.title('Ground Truth')
    plt.axis('off');
    plt.subplot(133);
    plt.imshow(a['out'].cpu().detach().numpy()[0][0]>0.7);
    plt.title('Segmentation Output')
    plt.axis('off');
    plt.savefig('./out/SegmentationOutput_' + f,bbox_inches='tight')
    img = a['out'].cpu().detach().numpy()[0][0] > 0.7
    img = img.astype(np.uint8)  # convert to an unsigned byte
    img *= 255
    # find size of original image
    img = cv2.resize(img, img_dims)
    cv2.imwrite("out/" + imgname + "_segout.jpg", img)