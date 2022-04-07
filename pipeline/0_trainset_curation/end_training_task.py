from sourceRdefs import endTrainingTask, mergeUpdateAnnotations

# move new training segments to data if they exist, delete all images in training task and call mergeUpdateAnnotations to merge new annotations
endTrainingTask(trainer="Rob",task_name="4-5-22_Classes1000")