from getFilterImageIDs import getFilterImageIDs
from createTrainingTask import createTrainingTask
from updateTrainingMetadata import updateTrainingMetadata

# merge annotations creates a new annotations.csv file using completed training tasks (NOTE - this removes the in_task column)
#updateTrainingMetadata()

# get image ids to move to training task, staying exclusive via not_in_train_data
#training_id_pool = getFilterImageIDs(contains_str="INATRANDOM",train_fields=["class","has_segment"],train_values=["dorsolateral",-1])
#createTrainingTask(trainer_name="Louis",task_name="10-4-22_DLSegs50",image_id_pool=training_id_pool,num_images=50)

#training_id_pool = getFilterImageIDs(contains_str="INATRANDOM",train_fields=["class","has_segment","in_active_task"],train_values=["dorsal",-1,-1])
#createTrainingTask(trainer_name="Camilla",task_name="10-4-22_LSegs50",image_id_pool=training_id_pool,num_images=50)

training_id_pool = getFilterImageIDs(contains_str="INATRANDOM",train_fields=["class","has_segment"],train_values=["lateral",-1])
createTrainingTask(trainer_name="Lauren",task_name="10-4-22_DSegs50",image_id_pool=training_id_pool,num_images=50)
