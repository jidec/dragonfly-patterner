from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models, datasets, transforms
import copy
import torch
import os
from os import path
from segdataset import SegmentationDataset

data_transforms = {
    # data augmentation and normalization for training
    # compose combines transforms
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224), # crop a random part of the image and resize it to size 224
        transforms.Resize([224,224]),
        transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
        transforms.ToTensor(), # convert a pil image or numpy ndarray to tensor
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # just normalization for validation
    'val': transforms.Compose([
        transforms.Resize(224), # resize image
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                progress=True)
model.classifier = DeepLabHead(2048, 1) # 1 mask output channel
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 10

data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

image_datasets = {
    x: SegmentationDataset(root= "data" / x,
                           transforms=data_transforms,
                           image_folder="Image",
                           mask_folder="Mask")
    for x in ['train', 'val']
}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1e10

# use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, num_epochs + 1):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['Train', 'Test']:
        if phase == 'Train':
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
            with torch.set_grad_enabled(phase == 'Train'):
                outputs = model(inputs)
                loss = criterion(outputs['out'], masks)
                y_pred = outputs['out'].data.cpu().numpy().ravel()
                y_true = masks.data.cpu().numpy().ravel()

                # backward + optimize only if in training phase
                if phase == 'Train':
                    loss.backward()
                    optimizer.step()
        epoch_loss = loss
        print('{} Loss: {:.4f}'.format(phase, loss))

        # deep copy the model
        if phase == 'Test' and loss < best_loss:
            best_loss = loss
            best_model_wts = copy.deepcopy(model.state_dict())

print('Lowest Loss: {:4f}'.format(best_loss))

# load best model weights
model.load_state_dict(best_model_wts)