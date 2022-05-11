from sourceRdefs import moveSelectImages,mergeUpdateAnnotations,copyClassImagesForTraining
import loadTrainClassModel
from torchvision import transforms
import AddGaussianNoise
import shutil

# set augments
data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([400,400]),
            transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
            transforms.ToTensor(), # convert a pil image or numpy ndarray to tensor
            AddGaussianNoise(0., 1.), # add Gaussian noise
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # just normalization for validation
        'test': transforms.Compose([
            transforms.Resize([400,400]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

## create dorsal_non classifier model
# move dorsal images to dorsal dir
copyClassImagesForTraining(class_col = "dorsal_lateral_bad", class_name = "dorsal",
                          to = "pylib/trainsets/dorsal_lateral_bad",
                          split_test_train = True, ntest = 15)
# move bad and lateral images to non dir
copyClassImagesForTraining(class_col = "dorsal_lateral_bad", class_name = "lateral", class_dir_override="non",
                          to = "pylib/trainsets/dorsal_lateral_bad",
                          split_test_train = True, ntest = 15)
copyClassImagesForTraining(class_col = "class", class_name = "bad", class_dir_override="non", from_ = "../data/random_images",
                          to = "../experiments/odo_view_classifier/dorsal_lateral_bad",
                          split_test_train = True, ntest = 15)

# load, train, and save model
loadTrainClassModel(data_dir="../experiments/odo_view_classifier/dorsal_non",
                    exp_dir="../experiments/odo_view_classifier/dorsal_non",
                    num_epochs=10, batch_size=4, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="dorsal_non")

shutil.rmtree('/path/to/your/dir/')

#----------
# Create lateral_non classifier model

# move lateral images to dorsal dir
moveAnnotationClassImages(class_col = "class", class_name = "lateral", from_ = "../data/random_images",
                          to = "../experiments/odo_view_classifier/dorsal_lateral_bad",
                          split_test_train = True, ntest = 5)
# move bad and dorsal images to non dir
moveAnnotationClassImages(class_col = "class", class_name = "dorsal", class_dir_override="non", from_ = "../data/random_images",
                          to = "../experiments/odo_view_classifier/dorsal_lateral_bad",
                          split_test_train = True, ntest = 5)
moveAnnotationClassImages(class_col = "class", class_name = "bad", class_dir_override="non", from_ = "../data/random_images",
                          to = "../experiments/odo_view_classifier/dorsal_lateral_bad",
                          split_test_train = True, ntest = 5)

# load, train, and save model
loadTrainClassModel(data_dir="../experiments/odo_view_classifier/lateral_non",
                    exp_dir="../experiments/odo_view_classifier/lateral_non",
                    num_epochs=10, batch_size=4, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="lateral_non")

#----------
# Create dorsal_perfect classifier model

# move normal dorsal images
moveAnnotationClassImages(class_col = "class", class_name = "dorsal", class_col2="is_perfect", class_name2="0", from_ = "../data/random_images",
                          to = "../experiments/odo_view_classifier/dorsal_perfect",
                          split_test_train = True, ntest = 5)
# move perfect dorsal images
moveAnnotationClassImages(class_col = "class", class_name = "dorsal", class_col2="is_perfect", class_name2="1", from_ = "../data/random_images",
                          to = "../experiments/odo_view_classifier/dorsal_perfect",
                          split_test_train = True, ntest = 5)

# load, train, and save model
loadTrainClassModel(data_dir="../experiments/odo_view_classifier/dorsal_perfect",
                    exp_dir="../experiments/odo_view_classifier/dorsal_perfect",
                    num_epochs=10, batch_size=4, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="dorsal_perfect")

#----------
# Create lateral_perfect classifier model

# move normal lateral images
moveAnnotationClassImages(class_col = "class", class_name = "lateral", class_col2="is_perfect", class_name2="0", from_ = "../data/random_images",
                          to = "../experiments/odo_view_classifier/lateral_perfect",
                          split_test_train = True, ntest = 5)
# move perfect lateral images
moveAnnotationClassImages(class_col = "class", class_name = "lateral", class_col2="is_perfect", class_name2="1", from_ = "../data/random_images",
                          to = "../experiments/odo_view_classifier/lateral_perfect",
                          split_test_train = True, ntest = 5)

# load, train, and save model
loadTrainClassModel(data_dir="../experiments/odo_view_classifier/lateral_perfect",
                    exp_dir="../experiments/odo_view_classifier/lateral_perfect",
                    num_epochs=10, batch_size=4, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="lateral_perfect")