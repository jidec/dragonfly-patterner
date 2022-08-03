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

# 0_data_prep
# merge records csvs and rename some cols
#
# write the genus list for use by downloader
writeiNatGenusList(inat_csv_name="inatdragonflyusa_records")
# download a genus
downloadiNatGenusImages(start_index=7,end_index=8,skip_records=True,skip_images=False)

# 0_trainset_curation
# create and complete training sets
# update annotations so all finished annotations are placed in data/train_metadata.csv
updateTrainingMetadata()

# 1_classification
# get ids of each class in training set only
dorsal_ids = getFilterImageIDs(train_fields=["class"],train_values=["dorsal"])
lateral_ids = getFilterImageIDs(train_fields=["class"],train_values=["lateral"])
dl_ids = getFilterImageIDs(train_fields=["class"],train_values=["dorsolateral"])
bad_ids = getFilterImageIDs(train_fields=["class"],train_values=["bad"])

# copy images by id to training dirs
copyImagesToTrainingDir(training_dir_name="4-class", image_ids=dorsal_ids, ntest = 150, class_dir_name="dorsal", proj_dir="../..")
copyImagesToTrainingDir("4-class", lateral_ids, 150, "lateral", proj_dir="../..")
copyImagesToTrainingDir("4-class", dl_ids, 150, "dorsolateral", proj_dir="../..")
copyImagesToTrainingDir("4-class", bad_ids, 150, "bad", proj_dir="../..")

# load, train, and save model
# still have to update this to match new paradigm
loadTrainClassModel(data_dir="../../data/other/training_dirs/4-class",
                    num_epochs=12, batch_size=3, num_workers=0,
                    model_name="4-class-loss1",
                    model_dir="../../data/ml_models")

# get ids not in train data
ids = getFilterImageIDs(not_in_train_data=True)
# infer classes
inferImageClasses(image_ids=ids, infer_colname= "dorsal_lateral_bad", infer_names= ["bad","dorsal","dorsolateral","lateral"],
                  model_name="4-class")

# 2_segmentation
# infer segment masks and save to data/masks
inferSegments(ids,model_location='../../data/ml_models/segmenter.pt',activation_threshold=0.7,show=False)
# extract segments using masks, modifying the raw masks according to params
extractHoneSegments(ids,bound=True,remove_islands=False,set_nonwhite_to_black=True,erode=True,erode_kernel_size=3,write=True)

# 3_discretization_compositing
colorDiscretize(ids,group_cluster_records_col="species")
inferClusterMorphs(ids,records_group_col="species",classes=["dorsal","lateral","dorsolateral"],cluster_image_type="pattern")



