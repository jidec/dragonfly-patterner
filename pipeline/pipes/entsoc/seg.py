from mergePreprocessRecords import mergePreprocessRecords
from writeiNatGenusList import writeiNatGenusList
from downloadiNatGenusImages import downloadiNatGenusImages
from updateTrainingMetadata import updateTrainingMetadata
from getFilterImageIDs import getFilterImageIDs
from copyImagesToTrainingDir import copyImagesToTrainingDir
from loadTrainClassModel import loadTrainClassModel
from inferImageClasses import inferImageClasses
from inferSegments import inferSegments
from extractHoneSegments import extractHoneSegments
from colorDiscretize import colorDiscretize
from inferClusterMorphs import inferClusterMorphs
import random

# get ids not in train data
ids = getFilterImageIDs(not_in_train_data=True,records_fields=["family"],records_values=["Aeshnidae"],# infer_fields=["conf_infers5"],infer_values="dorsal",
                        proj_dir="../../..")
ids = ids[11000:len(ids)]
# infer segments
inferSegments(image_ids=ids, model_name="segmenter_grey_contrast_sharp_b6_bce15_101",
              greyscale=True, image_size=344, show=False, proj_dir="../../..")
