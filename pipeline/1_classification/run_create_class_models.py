#from sourceRdefs import moveSelectImages,mergeUpdateAnnotations,copyClassImagesForTraining,copyClassImagesToTrainingDirs
from updateTrainingMetadata import updateTrainingMetadata
from loadTrainClassModel import loadTrainClassModel
from torchvision import transforms
from AddGaussianNoise import AddGaussianNoise
import shutil
from getFilterImageIDs import getFilterImageIDs
from copyImagesToTrainingDir import copyImagesToTrainingDir

# 87% lateral #90% dorsal # 94% dl

# update annotations so all finished annotations from trainset_tasks are known
#updateTrainingMetadata()

# specify size to reduce image to
image_pixels = 344

# set image augments
data_transforms = {
        'default': transforms.Compose([
            transforms.Resize([image_pixels,image_pixels]),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'train': transforms.Compose([
            transforms.Resize([image_pixels,image_pixels]),
            transforms.RandomAutocontrast(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
            transforms.ToTensor(), # convert a pil image or numpy ndarray to tensor
        ]),
    }

# load, train, and save model
# still have to update this to match new paradigm
loadTrainClassModel(training_dir_name="4-class-backup",
                    num_epochs=20, batch_size=6, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="4-class-aug") #loss_matrix_name="dfly_views_loss_mat")

# load, train, and save model
# still have to update this to match new paradigm
loadTrainClassModel(training_dir_name="3-class",
                    num_epochs=20, batch_size=6, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="3-class-grey-aug") #loss_matrix_name="dfly_views_loss_mat")

# load, train, and save model
# still have to update this to match new paradigm
loadTrainClassModel(training_dir_name="3-class-color",
                    num_epochs=20, batch_size=6, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="3-class-aug") #loss_matrix_name="dfly_views_loss_mat")

# next step - make sure inferences include weights and test visualizing weight based labels