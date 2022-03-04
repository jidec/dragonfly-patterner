from pathlib import Path
import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data

from getDataloaders import getDataloaders
from trainModel import trainModel

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
from SegmentationDataset import SegmentationDataset

@click.command()
@click.option("--data_dir",
              required=True,
              help="Specify the data directory.")
@click.option("--exp_dir",
              required=True,
              help="Specify the experiment directory.")
@click.option("--num_epochs",
              default=25,
              type=int,
              help="Specify the number of epochs you want to run the experiment for.")
@click.option("--batch_size",
              default=4,
              type=int,
              help="Specify the batch size for the dataloader.")
@click.option("--num_workers",
              default=4,
              type=int,
              help="Specify the number of workers for the dataloader.")

def loadTrainSegModel(data_dir, exp_directory, num_epochs, batch_size, num_workers):

    # download the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset
    # TODO- allow specifying a model
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)

    # set the model head
    # change output channel size to greater than 1 for multichannel segmentation
    model.classifier = DeepLabHead(2048, 1)

    # put the model in training mode
    model.train()
    data_dir = Path(data_dir)

    # create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # specify the loss function
    # TODO - allow specifying a loss function in cmd
    criterion = torch.nn.MSELoss(reduction='mean')

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

    data_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224), # crop a random part of the image and resize it to size 224
        transforms.Resize([344, 344]),
        # transforms.Resize([448, 448]),
        # transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
        transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {
        x: SegmentationDataset(root=Path(data_dir) / x,
                               transforms=data_transforms,
                               image_folder="image",
                               mask_folder="mask")
        for x in ['train', 'test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=4)
        for x in ['train', 'test']
    }

    trainModel(model,
               criterion,
               dataloaders,
               optimizer,
               bpath=exp_directory,
               metrics=metrics,
               num_epochs=num_epochs)

    # save the trained model
    torch.save(model, exp_directory / 'segmodel_weights.pt')

if __name__ == "__main__":
    loadTrainSegModel()
