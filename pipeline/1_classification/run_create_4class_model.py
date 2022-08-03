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
updateTrainingMetadata()

# specify size to reduce image to
image_pixels = 256

# set image augments
data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([image_pixels,image_pixels]),
            transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
            #transforms.RandomVerticalFlip(),
            #transforms.RandomRotation(degrees=(0,360)),
            #transforms.GaussianBlur(kernel_size=7),
            #transforms.ColorJitter(brightness=(0.25,0.75),contrast=(0.25,0.75),saturation=(0.25,0.75)),
            transforms.ToTensor(), # convert a pil image or numpy ndarray to tensor
            #AddGaussianNoise(0., 1.), # add Gaussian noise
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # just normalization for validation
        'test': transforms.Compose([
            transforms.Resize([image_pixels,image_pixels]),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

dorsal_ids = getFilterImageIDs(train_fields=["class"],train_values=["dorsal"])
print(len(dorsal_ids))
lateral_ids = getFilterImageIDs(train_fields=["class"],train_values=["lateral"])
print(len(lateral_ids))
dl_ids = getFilterImageIDs(train_fields=["class"],train_values=["dorsolateral"])
print(len(dl_ids))
bad_ids = getFilterImageIDs(train_fields=["class"],train_values=["bad"])
print(len(bad_ids))

#copyImagesToTrainingDir(training_dir_name="4-class", image_ids=dorsal_ids, ntest = 150, class_dir_name="dorsal", proj_dir="../..")
#copyImagesToTrainingDir("4-class", lateral_ids, 150, "lateral", proj_dir="../..")
#copyImagesToTrainingDir("4-class", dl_ids, 150, "dorsolateral", proj_dir="../..")
#copyImagesToTrainingDir("4-class", bad_ids, 150, "bad", proj_dir="../..")


# load, train, and save model
# still have to update this to match new paradigm
loadTrainClassModel(training_dir_name="4_class",
                    num_epochs=12, batch_size=3, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="4-class-loss1",
                    model_dir="../../data/ml_models",loss_matrix_name="dfly_views_loss_mat")

# remove copied training images from the temporary training dir
# shutil.rmtree(path="../../data/training_dirs/4-class")