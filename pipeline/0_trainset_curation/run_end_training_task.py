from endTrainingTask import endTrainingTask
from invertMasks import invertMasks

# move new training segments to data if they exist, delete all images in training task and call mergeUpdateAnnotations to merge new annotations
endTrainingTask(trainer_names=["Jacob","Jacob","Jacob"],task_names=["8-30-22_DLSegments75","8-30-22_DorsalSegments75","8-30-22_LatSegments75"],proj_dir="E:/dragonfly-patterner")