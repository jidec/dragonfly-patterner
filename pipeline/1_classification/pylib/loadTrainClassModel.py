from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import os
from trainClassifier import trainClassifier
from torchsampler import ImbalancedDatasetSampler #pip install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip

# pretrained model can be models.resnet18 or models.inception_v3
def loadTrainClassModel(data_dir, num_epochs, batch_size, num_workers, data_transforms, model_name, model_dir,
                        pretrained_model=models.resnet18(pretrained=True), criterion=nn.modules.loss.CrossEntropyLoss()):

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, #sampler=ImbalancedDatasetSampler(image_datasets[x])
                                                 shuffle=True, num_workers=num_workers) #shuffle = True
                  for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training model on " + str(device))

    model = pretrained_model
    num_ftrs = model.fc.in_features
    # generalized to nn.Linear(num_ftrs, len(class_names)).
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.aux_logits = False

    #model = models.vgg16(pretrained=True)
    #num_ftrs = model.fc.in_features
    #num_ftrs = model.classifier[0].in_features
    #model.fc = nn.Linear(num_ftrs,len(class_names))
    #model.classifier[0] = nn.Linear(num_ftrs, len(class_names))
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.01) #lr = 0.001 momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    model = trainClassifier(model=model, dataloaders=dataloaders,dataset_sizes=dataset_sizes, criterion=criterion,
                             optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=num_epochs,class_names=class_names)
    
    torch.save(model, model_dir + "/" + model_name + ".pt")