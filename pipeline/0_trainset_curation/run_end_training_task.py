from endTrainingTask import endTrainingTask
from invertMasks import invertMasks

# move new training segments to data if they exist, delete all images in training task and call mergeUpdateAnnotations to merge new annotations
endTrainingTask(trainer_names=["Jacob","Ana","Louis"],task_names=["X,X,X"],proj_dir="E:/dragonfly-patterner")