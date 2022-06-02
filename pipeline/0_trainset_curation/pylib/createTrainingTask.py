from random import sample
from copyImages import copyImages
import os

def createTrainingTask(trainer_name,task_name,image_id_pool,num_images,proj_root="../.."):
    sample_images = sample(image_id_pool,num_images)
    print("Sampled " + str(num_images) + " images from provided pool of " + str(len(image_id_pool)))

    # make the trainer folder if it doesn't exist
    trainer_path = proj_root + "/trainset_tasks/" + trainer_name
    if not os.path.exists(trainer_path):
        os.mkdir(trainer_path)

    # make the task dir
    task_path = trainer_path + "/" + task_name
    if not os.path.exists(task_path):
        os.mkdir(task_path)
    print("Created new training task directory")

    # copy images
    copyImages(source=proj_root + "/data/all_images",dest= task_path,ids=sample_images)
    print("Copied images to training task - finished!")