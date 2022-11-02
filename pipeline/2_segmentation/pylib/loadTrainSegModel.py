from pathlib import Path
import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
from trainModel import trainModel
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torch.utils.data import DataLoader
from SegmentationDataset import SegmentationDataset
from torchvision import transforms
import os

def loadTrainSegModel(training_dir_name, num_epochs, batch_size, num_workers, model_name, data_transforms=None,
                      criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(15)), resnet_type="deeplabv3", resnet_size="50", pretrained=True,
                      print_steps=True, print_details=False, proj_dir="../.."):
    """
        Load and train a segmentation model

        :param str training_dir_name: the name of the training dir within data/other/training_dirs
        :param int num_epochs: the number of epochs to train for
        :param int batch_size: the number of images to train on per batch
        :param int num_workers: the number of workers
        :param str model_name: what to call the outputted model.pt
        :param Dictionary data_transforms: instructions for transforming the data, see default example
    """
    if print_steps: print("Started loadTrainSegModel")

    if data_transforms is None:
        data_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(224), # crop a random part of the image and resize it to size 224
            transforms.Resize([344, 344]),
            # transforms.Resize([448, 448]),
            transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            #transforms.Grayscale()
        ])
        if print_steps: print("Loaded default transforms")

    # download the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset
    # TODO- allow specifying a model
    if resnet_size == "50":
        if resnet_type == "deeplabv3":
            model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained,
                                                            progress=True)
        elif resnet_type == "maskrcnn":
            model = models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
    elif resnet_size == "101":
        model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained,
                                                       progress=True)
    # set the model head
    # change output channel size to greater than 1 for multichannel segmentation
    model.classifier = DeepLabHead(2048, 1)

    # put the model in training mode
    model.train()

    # create the experiment directory if not present
    exp_directory = proj_dir + "/data/ml_models/logs"
    # not exp_directory.exists():
    #    exp_directory.mkdir()

    # specify the loss function
    # criterion = torch.nn.MSELoss(reduction='mean')

    # specify the optimizer with a learning rate
    # TODO - allow specifying a learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    # create the dataloader
    # dataloaders = getDataloaders(
    #     data_directory, batch_size=batch_size)
    # _ = trainModel(model,
    #                 criterion,
    #                 dataloaders,
    #                 optimizer,
    #                 bpath=exp_directory,
    #                 metrics=metrics,
    #                 num_epochs=num_epochs)

    image_datasets = {
        x: SegmentationDataset(root=Path(proj_dir + "/data/other/training_dirs/" + training_dir_name) / x,
                               transforms=data_transforms,
                               image_folder="Image",
                               mask_folder="Mask")
        for x in ['Train', 'Test']
    }
    if print_steps: print("Created datasets object")

    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers)
        for x in ['Train', 'Test']
    }
    if print_steps: print("Created dataloaders object")

    if print_steps: print("Training model...")
    trainModel(model,
               criterion,
               dataloaders,
               optimizer,
               bpath=exp_directory,
               metrics=metrics,
               num_epochs=num_epochs,
               name=model_name)

    # save the trained model
    dest = proj_dir + '/data/ml_models/' + model_name + '.pt'
    torch.save(model, dest)
    if print_steps: print("Saved model to " + dest)
