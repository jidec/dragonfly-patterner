from loadTrainClassModel import loadTrainClassModel
from torchvision import transforms
from inferImages import inferImages

image_dir = '../../data/random_images'
model_location = 'dorsal_model.pt'

# load and train
inferImages(image_dir=image_dir, model_location=model_location,image_size=344,show=True)