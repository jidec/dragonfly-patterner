from random import sample
from copyImages import copyImages
import os
import pandas as pd
import shutil

def endTrainingTask(trainer_names,task_names, proj_dir="../.."):
    """
        End a training task

        :param str trainer_names: the names of the trainers to end a task for
        :param str task_names: the names of the tasks to end
        :param str proj_root: the location of the patterner project folder
    """

    for trainer_name, task_name in tuple(zip(trainer_names, task_names)):

        # load image file names
        source = proj_dir + "/trainset_tasks/" + trainer_name + "/" + task_name
        file_names = os.listdir(source)
        image_filenames = [k for k in file_names if '.jpg' in k]

        # get mask info
        mask_files = [k for k in file_names if '_mask' in k]
        has_masks = len(mask_files) > 0

        # if has masks, it is a segmentation task so copy those to train_masks first
        if has_masks:
            source_files = [source + "/" + s for s in mask_files]
            dest_files = [proj_dir + "/data/masks/train_masks/" + s for s in mask_files]
            for source, dest in tuple(zip(source_files, dest_files)):
                shutil.copy(source, dest)

            # save segmented ids as csv
            ids = [s.removesuffix(".jpg") for s in image_filenames]
            ones = [1] * len(ids)
            df = pd.DataFrame(data={'file': ids, 'has_segment': ones})
            #ids = pd.Series(ids)
            source = proj_dir + "/trainset_tasks/" + trainer_name + "/" + task_name
            df.to_csv(source + "/segmented_ids.csv")

        source = proj_dir + "/trainset_tasks/" + trainer_name + "/" + task_name
        # remove every image file
        for file in image_filenames:
            os.remove(source + "/" + file)