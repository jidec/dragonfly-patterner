from torchvision import datasets, models, transforms
import torch.nn as nn
import torch

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 3) #2

model.load_state_dict(torch.load("model.pt"))

# for each file in all_images
#input = numpy...
out = model(input)

#numpy.savetxt('output.csv', arr)

