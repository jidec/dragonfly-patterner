from random import sample
from copyImages import copyImages
import os
import pandas as pd
import numpy as np

def createTrainingTask(trainer_name,task_name,image_id_pool,num_images,proj_dir="../.."):
    """
        Create a training task

        :param str trainer_name: the name of the trainer to add a task for
        :param str task_name: the name of the new task
        :param list<str> image_id_pool: a set of image IDs to draw from
        :param int num_images: the number of images to draw from the pool
        :param str proj_root: the location of the patterner project folder
    """

    sample_image_ids = sample(image_id_pool,num_images)
    print("Sampled " + str(num_images) + " images from provided pool of " + str(len(image_id_pool)) + "...")

    # make the trainer folder if it doesn't exist
    trainer_path = proj_dir + "/trainset_tasks/" + trainer_name
    if not os.path.exists(trainer_path):
        os.mkdir(trainer_path)

    # make the task dir
    task_path = trainer_path + "/" + task_name
    if not os.path.exists(task_path):
        os.mkdir(task_path)
    print("Created new training task directory...")

    # add image IDs to in_task column in train_metadata
    train_metadata = pd.read_csv(proj_dir + "/data/train_metadata.csv")
    id_series = pd.Series([s + ".jpg" for s in sample_image_ids], name="file")
    one_series = pd.Series(np.repeat(1,len(sample_image_ids)),name="in_task")
    #df = pd.concat([id_series, one_series],axis=0)
    df = pd.merge(id_series, one_series, right_index=True,
                  left_index=True)

    train_metadata = pd.concat([train_metadata,df])
    train_metadata.fillna(int(-1), inplace=True)
    #print(train_metadata)
    train_metadata.to_csv(proj_dir + "/data/train_metadata.csv")
    print("Added training samples to the in_task column to maintain exclusivity via the not_in_train_data parameter of getFilterImageIDs until updateTrainingMetadata is called again...")

    # copy images
    copyImages(source=proj_dir + "/data/all_images",dest= task_path,ids=sample_image_ids)
    print("Copied images to training task - finished!")