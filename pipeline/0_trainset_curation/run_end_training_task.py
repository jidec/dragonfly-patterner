from endTrainingTask import endTrainingTask

# move new training segments to data if they exist, delete all images in training task and call mergeUpdateAnnotations to merge new annotations
endTrainingTask(trainer_names=["Jacob","Jacob"],task_names=["6-3-22_Classes1000","5-23-22_Classes500"],proj_dir="E:/dragonfly-patterner")

endTrainingTask(trainer_names=["Ana","Ana","Ana"],task_names=["6-3-22_Classes1000","5-23-22_Classes500","6-3-22_LateralSegments26"],proj_dir="E:/dragonfly-patterner")

endTrainingTask(trainer_names=["Louis","Louis","Louis"],task_names=["6-3-22_Classes1000","5-23-22_Classes500","6-3-22_DorsolateralSegments50"],proj_dir="E:/dragonfly-patterner")