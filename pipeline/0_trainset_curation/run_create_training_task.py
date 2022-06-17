from getFilterImageIDs import getFilterImageIDs
from createTrainingTask import createTrainingTask
from updateTrainingMetadata import updateTrainingMetadata

# merge annotations creates a new annotations.csv file using completed training tasks (NOTE - this removes the in_task column)
updateTrainingMetadata()

# get image ids to move to training task, staying exclusive via not_in_train_data
training_id_pool = getFilterImageIDs(contains_str="INATRANDOM",train_fields=["class","has_segment"],train_values=["dorsal",-1])
createTrainingTask(trainer_name="Ana",task_name="6-17-22_Segments75",image_id_pool=training_id_pool,num_images=75)

training_id_pool = getFilterImageIDs(contains_str="INATRANDOM",train_fields=["class","has_segment","in_task"],train_values=["dorsolateral",-1,-1])
createTrainingTask(trainer_name="Jacob",task_name="6-17-22_Segments75",image_id_pool=training_id_pool,num_images=75)

training_id_pool = getFilterImageIDs(contains_str="INATRANDOM",train_fields=["class","has_segment","in_task"],train_values=["lateral",-1,-1])
createTrainingTask(trainer_name="Louis",task_name="6-17-22_Segments75",image_id_pool=training_id_pool,num_images=75)
