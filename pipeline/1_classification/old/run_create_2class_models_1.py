from sourceRdefs import moveSelectImages,mergeUpdateAnnotations,copyClassImagesForTraining
from loadTrainClassModel import loadTrainClassModel
from torchvision import transforms
from AddGaussianNoise import AddGaussianNoise
import shutil
import os
from torchvision import models

# set image augments
data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([400,400]),
            transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
            transforms.RandomRotation(degrees=(0,360)),
            transforms.GaussianBlur(kernel_size=7),
            transforms.ColorJitter(brightness=(0.25,0.75),contrast=(0.25,0.75),saturation=(0.25,0.75)),
            transforms.ToTensor(), # convert a pil image or numpy ndarray to tensor
            #AddGaussianNoise(0., 1.), # add Gaussian noise
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # just normalization for validation
        'test': transforms.Compose([
            transforms.Resize([400,400]),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

# move dorsal images to dorsal-lateral dir
copyClassImagesForTraining(class_col = "dorsal_lateral_bad", class_name = "dorsal",
                          to = "pylib/trainsets/bad_non",
                          split_test_train = True, ntest = 15,class_dir_override="non")

# move lateral images to dorsal-lateral dir
copyClassImagesForTraining(class_col = "dorsal_lateral_bad", class_name = "lateral",
                          to = "pylib/trainsets/bad_non",
                          split_test_train = True, ntest = 15,class_dir_override="non")

# move lateral images to dorsal-lateral dir
copyClassImagesForTraining(class_col = "dorsal_lateral_bad", class_name = "bad",
                          to = "pylib/trainsets/bad_non",
                          split_test_train = True, ntest = 30)

# load, train, and save model
loadTrainClassModel(data_dir="pylib/trainsets/bad_non",
                    num_epochs=15, batch_size=4, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="bad_non",
                    model_dir="pylib/models",
                    pretrained_model=models.resnet101(pretrained=True))

# remove temporary training folders
for train_dir in os.listdir("pylib/trainsets"):
    shutil.rmtree(train_dir)