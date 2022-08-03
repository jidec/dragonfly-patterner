import os
import pandas as pd
from os.path import exists
from imageIDToRecordID import imageIDToRecordID

def getFilterImageIDs(start_ids=None,records_fields=[],records_values=[], in_records_data=False,
                      train_fields=[], train_values=[], not_in_train_data=False,
                      infer_fields=[],infer_values=[], not_in_inference_data=False,
                      print_steps=True,contains_str=None,proj_dir="../.."):
    """
        Get image ids from images, records, training_metadata, and inference csvs in the data folder and filter them using various criteria
        Used to feed image ids to move images, create training tasks, and more
    """

    # get image_ids
    if start_ids == None:
        image_ids = [s.replace('.jpg', '') for s in os.listdir(proj_dir + "/data/all_images")]
        if print_steps: print("Retrieved " + str(len(image_ids)) + " ids from images...")
    else:
        image_ids = start_ids
        if print_steps: print("Using start IDs")

    # read in records
    records = pd.read_csv(proj_dir + "/data/records.csv")

    # filter for records matching each field value combination
    for field, value in tuple(zip(records_fields, records_values)):
        records = records[records[field] == value]

    # add all records matching
    records_ids = list(records['recordID'])
    if print_steps: print("Retrieved/filtered " + str(len(records_ids)) + " ids from records...")

    if len(records_fields) > 0 or in_records_data == True:
        image_record_ids = imageIDToRecordID(image_ids)
        if print_steps: print("Converted image ids to records ids")

        # convert to a series, get boolean indices of that series, then convert back to a list
        image_ids_series = pd.Series(image_ids)
        image_ids_series = image_ids_series[pd.Series(image_record_ids).isin(records_ids)]

        image_ids = list(image_ids_series)
        if print_steps: print("Filtered to " + str(len(image_ids)) + " for IDs in records data...")

    # read in training metadata
    if exists(proj_dir + "/data/train_metadata.csv") and len(train_fields) > 0:
        train_metadata = pd.read_csv(proj_dir + "/data/train_metadata.csv")

        # filter for records matching every value in every respective field
        for field, value in tuple(zip(train_fields,train_values)):
            train_metadata = train_metadata[train_metadata[field] == value]
        train_ids = train_metadata['imageID']
        # match image ids to train ids
        image_ids = list(set(image_ids).intersection(train_ids))
        if print_steps: print("Filtered to " + str(len(image_ids)) + " using training...")

    if not_in_train_data == True:
        train_metadata = pd.read_csv(proj_dir + "/data/train_metadata.csv")
        train_ids = train_metadata['imageID']
        image_ids = list(set(image_ids).difference(train_ids))
        if print_steps: print("Filtered to " + str(len(image_ids)) + " for records not in train data...")

    # read in inference data
    if exists(proj_dir + "/data/inferences.csv") and len(infer_fields) > 0:
        inferences = pd.read_csv(proj_dir + "/data/inferences.csv")
        # filter for records matching every value in every respective field
        for field, value in tuple(zip(infer_fields, infer_values)):
            inferences = inferences[inferences[field] == value]
        infer_ids = inferences['imageID']
        # match image ids to infer ids
        image_ids = list(set(image_ids).intersection(infer_ids))
        if print_steps: print("Filtered to " + str(len(image_ids)) + " using inferences...")

    if not_in_inference_data == True:
        inferences = pd.read_csv(proj_dir + "/data/inferences.csv")
        infer_ids = inferences['imageID']
        image_ids = list(set(image_ids).difference(infer_ids))
        if print_steps: print("Filtered to " + str(len(image_ids)) + " for records not in train data...")

    # get intersection of image_ids, infer_ids and train_ids - this is all the image ids that match the train and infer values
    #ids = image_ids
    #if
    #ids = list(set(infer_ids).intersection(train_ids))
    #ids = list(set(image_ids).intersection(ids))

    # filter for string contains
    if contains_str != None:
        image_ids = [s for s in image_ids if contains_str in s]
        if print_steps: print("Filtered to " + str(len(image_ids)) + " using the contains string...")

    if print_steps: print("Done filtering" + "\n")

    return(image_ids)

