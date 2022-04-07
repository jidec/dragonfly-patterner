from torch.utils.data import DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models, transforms
import copy
import torch
from segmentation_data_class import SegmentationDataset
from datahandler import get_dataloader_sep_folder
from pathlib import Path
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# create transforms to apply to train and test sets
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([448,448]),
        transforms.ToTensor(), # convert a pil image or numpy ndarray to tensor
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
    ]),
    # just normalization for validation
    'val': transforms.Compose([
        transforms.Resize([448,448]), # resize image
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# import deeplabv3 model and add 1 mask output channel
model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                progress=True)
model.classifier = DeepLabHead(2048, 1)

# set loss function
criterion = torch.nn.MSELoss()
#criterion = torch.nn.CrossEntropyLoss()

# set optimizer
optimizer = torch.optim.Adadelta(model.parameters(),lr=0.0001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# set num epochs
num_epochs = 3

# set data dir containing train/Mask, train/Image, val/Mask, val/Image
data_dir = 'data'

# create dict of two segmentation datasets, one for train and one for test
image_datasets = {
    x: SegmentationDataset(root=Path(data_dir) / x,
                           transforms=data_transforms,
                           image_folder="Image",
                           mask_folder="Mask")
    for x in ['train', 'val']
}

# create dataloaders
dataloaders = {
    x: DataLoader(image_datasets[x],
                  batch_size=4,
                  shuffle=True,
                  num_workers=0)
    for x in ['train', 'val']
}

dataloaders = get_dataloader_sep_folder(data_dir, batch_size=8)

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


# get a batch of training data
inputs = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs['image'])
img = inputs['image']

imshow(out) #title=[class_names[x] for x in classes])

# Make a grid from batch
out = torchvision.utils.make_grid(inputs['mask'])

imshow(out) #title=[class_names[x] for x in classes])

# init best model weights as copy of model state
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1e10

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, num_epochs + 1):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-' * 10)
    # Each epoch has a training and validation phase

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        # Iterate over data.
        for sample in iter(dataloaders[phase]):
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs['out'], masks)
                y_pred = outputs['out'].data.cpu().numpy().ravel()
                y_true = masks.data.cpu().numpy().ravel()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
        epoch_loss = loss
        print("Loss: " + str(epoch_loss))

        # deep copy the model
        if phase == 'val' and loss < best_loss:
            best_loss = loss
            best_model_wts = copy.deepcopy(model.state_dict())

print('Lowest Loss: {:4f}'.format(best_loss))

# load best model weights
model.load_state_dict(best_model_wts)
model.eval()

torch.save(model.state_dict(), "segmodel.pt")

model.eval()
inputs = next(iter(dataloaders['val']))
out = torchvision.utils.make_grid(inputs['mask'])
imshow(out)

test = model(inputs)

out = torchvision.utils.make_grid(inputs['image'])
imshow(out)
