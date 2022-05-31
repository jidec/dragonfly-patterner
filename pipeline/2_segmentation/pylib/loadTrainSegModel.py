from pathlib import Path
import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
from trainModel import trainModel
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torch.utils.data import DataLoader
from SegmentationDataset import SegmentationDataset

def loadTrainSegModel(data_dir, num_epochs, batch_size, num_workers, data_transforms, model_name, model_dir, export_dir):

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
    exp_directory = Path(export_dir)
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

    image_datasets = {
        x: SegmentationDataset(root=Path(data_dir) / x,
                               transforms=data_transforms,
                               image_folder="Image",
                               mask_folder="Mask")
        for x in ['Train', 'Test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers)
        for x in ['Train', 'Test']
    }

    trainModel(model,
               criterion,
               dataloaders,
               optimizer,
               bpath=exp_directory,
               metrics=metrics,
               num_epochs=num_epochs)

    # save the trained model
    torch.save(model, model_dir + '/' + model_name + '.pt')

if __name__ == "__main__":
    loadTrainSegModel()
