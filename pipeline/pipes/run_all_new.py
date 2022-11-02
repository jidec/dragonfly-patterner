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
ids = getFilterImageIDs(not_in_train_data=True,records_fields=["family"],records_values=["Aeshnidae"])

# infer classes
inferImageClasses(image_ids=ids, infer_colname="dorsal_lateral_bad", infer_names= ["bad","dorsal","lateral"],
                  model_name="3-class-aug",show=False)