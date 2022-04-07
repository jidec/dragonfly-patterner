from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import numpy as np
import matplotlib.pyplot as plt
import cv2

# import deeplabv3 model and add 1 mask output channel
model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                progress=True)
model.classifier = DeepLabHead(2048, 1)

model.load_state_dict(torch.load("segmodel.pt"))

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated

# for each file in all_images
#input = numpy...
# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model.eval()
img = cv2.imread('data/val/Image/1088260_874.jpg')
img = np.array(img)
img = cv2.resize(img,(448,448))
img = torch.from_numpy(img)
img.to(device)
out = model(img)



#numpy.savetxt('output.csv', arr)

