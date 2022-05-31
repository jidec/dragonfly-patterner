from sourceRdefs import endTrainingTask, mergeUpdateAnnotations

# move new training segments to data if they exist, delete all images in training task and call mergeUpdateAnnotations to merge new annotations
endTrainingTask(trainer="Jacob",task_name="50_Segments",proj_root="G:/ant-patterner")