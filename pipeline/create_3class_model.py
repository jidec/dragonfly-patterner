from sourceRdefs import moveSelectImages,mergeAnnotations,moveAnnotationClassImages
import loadTrainClassModel
from torchvision import transforms
import AddGaussianNoise

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

# move dorsal, lateral, and bad images to training set
moveAnnotationClassImages(class_col = "class", class_name = "dorsal", from_ = "../data/random_images",
                          to = "../experiments/odo_view_classifier/dorsal_lateral_bad",
                          split_test_train = True, ntest = 5)
moveAnnotationClassImages(class_col = "class", class_name = "lateral", from_ = "../data/random_images",
                          to = "../experiments/odo_view_classifier/dorsal_lateral_bad",
                          split_test_train = True, ntest = 5)
moveAnnotationClassImages(class_col = "class", class_name = "bad", from_ = "../data/random_images",
                          to = "../experiments/odo_view_classifier/dorsal_lateral_bad",
                          split_test_train = True, ntest = 5)

# load, train, and save model
loadTrainClassModel(data_dir="../experiments/odo_view_classifier/dorsal_lateral_bad",
                    exp_dir="../experiments/odo_view_classifier/dorsal_lateral_bad",
                    num_epochs=10, batch_size=4, num_workers=0,
                    data_transforms= data_transforms,
                    model_name="dorsal_lateral_bad")

# view some inferences 