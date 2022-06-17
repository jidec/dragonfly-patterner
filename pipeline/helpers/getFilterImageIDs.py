import os
import pandas as pd
from os.path import exists

def getFilterImageIDs(records_fields=[],records_values=[], in_records_data=False,
                      train_fields=[], train_values=[], not_in_train_data=False,
                      infer_fields=[],infer_values=[],contains_str=None,proj_dir="../.."):
    """
        Get image ids from images, records, training_metadata, and inference csvs in the data folder and filter them using various criteria
        Used to feed image ids to move images, create training tasks, and more
    """

    # get image_ids
    image_ids = [s.replace('.jpg', '') for s in os.listdir(proj_dir + "/data/all_images")]
    print("Retrieved " + str(len(image_ids)) + " ids from images...")

    # read in records
    records_files = [x for x in os.listdir(proj_dir + "/data") if "_records" in x]
    records_ids = []
    # for each records file
    for file in records_files:
        # read in records
        records = pd.read_csv(proj_dir + "/data/" + file)
        # for each field and respective value
        for field, value in tuple(zip(records_fields, records_values)):
            # filter for records matching this
            records = records[records[field] == value]
        # add all records matching
        records_ids = records_ids + list(records['recordID'])
    # records_ids is now a list containing all recordIDs from all record files matching the fields and values
    print("Retrieved " + str(len(records_ids)) + " ids from records...")

    # by the naming paradigm, the 0th split will be the source i.e. INAT,
    # 1st will be imageID i.e 3231, 2nd will be record extension i.e. -d, -1, -2
    # convert ids to match record ids
    if len(records_fields) > 0 or in_records_data == True:
        stripped_ids = [s.split('-')[1] for s in image_ids]
        # print("Stripped ids for records match")

        # get boolean indices of all IDs where stripped ids match filtered records ids
        isin_bools = pd.Series(stripped_ids).isin(records_ids)

        # convert to a series, get boolean indices of that series, then convert back to a list
        image_ids = list(pd.Series(image_ids, index=isin_bools))
        print("Filtered to " + str(len(image_ids)) + " for IDs in records data...")

    # read in training metadata
    if exists(proj_dir + "/data/train_metadata.csv") and len(train_fields) > 0:
        train_metadata = pd.read_csv(proj_dir + "/data/train_metadata.csv")

        # filter for records matching every value in every respective field
        for field, value in tuple(zip(train_fields,train_values)):
            train_metadata = train_metadata[train_metadata[field] == value]
        train_ids = train_metadata['imageID']
        # match image ids to train ids
        image_ids = list(set(image_ids).intersection(train_ids))
        print("Filtered to " + str(len(image_ids)) + " using training...")

    if not_in_train_data == True:
        train_metadata = pd.read_csv(proj_dir + "/data/train_metadata.csv")
        train_ids = train_metadata['imageID']
        image_ids = list(set(image_ids).difference(train_ids))
        print("Filtered to " + str(len(image_ids)) + " for records not in train data...")

    # read in inference data
    if exists(proj_dir + "/data/inferences.csv") and len(infer_fields) > 0:
        inferences = pd.read_csv(proj_dir + "/data/inferences.csv")
        # filter for records matching every value in every respective field
        for field, value in tuple(zip(infer_fields, infer_values)):
            inferences = inferences[inferences[field] == value]
        infer_ids = inferences['imageID']
        # match image ids to infer ids
        image_ids = list(set(image_ids).intersection(infer_ids))
        print("Filtered to " + str(len(image_ids)) + " using inferences...")

    # get intersection of image_ids, infer_ids and train_ids - this is all the image ids that match the train and infer values
    #ids = image_ids
    #if
    #ids = list(set(infer_ids).intersection(train_ids))
    #ids = list(set(image_ids).intersection(ids))

    # filter for string contains
    if contains_str != None:
        image_ids = [s for s in image_ids if contains_str in s]
        print("Filtered to " + str(len(image_ids)) + " using the contains string...")

    print("Done filtering - finished!")
    return(image_ids)

