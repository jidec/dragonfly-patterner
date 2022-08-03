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

def loadTrainClassModel(training_dir_name, num_epochs, batch_size, num_workers, model_name, data_transforms=None,
                        pretrained_model=models.resnet18(pretrained=True), criterion=nn.modules.loss.CrossEntropyLoss(),loss_matrix_name=None,
                        print_steps=False, proj_dir = "../.."):
    """
        Load and train a classification model

        :param str training_dir_name: the name of the training dir within data/other/training_dirs
        :param int num_epochs: the number of epochs to train for
        :param int batch_size: the number of images to train on per batch
        :param int num_workers: the number of workers
        :param str model_name: what to call the outputted model.pt
        :param Dictionary data_transforms: instructions for transforming the data, see default example
        :param pretrained_model: the pretrained model, either models.resnet18(pretrained=True) or models.inception_v3(pretrained=True)
    """

    # set default transforms
    if data_transforms is None:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),  # horizontally flip with 50% prob
                transforms.ToTensor(),
            ]),
            # just normalization for validation
            'test': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
            ]),
        }

    # create image datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(proj_dir + "/data/other/training_dirs/" + training_dir_name, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
    # create dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, #sampler=ImbalancedDatasetSampler(image_datasets[x])
                                                 shuffle=True, num_workers=num_workers) #shuffle = True
                  for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    if print_steps: print("Created datasets and dataloaders using params")

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
                             optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=num_epochs,class_names=class_names,loss_matrix_name=loss_matrix_name)

    model_path = proj_dir + "/data/ml_models/" + model_name + ".pt"
    torch.save(model, model_path)
    if print_steps: print("Saved model to " + model_path)