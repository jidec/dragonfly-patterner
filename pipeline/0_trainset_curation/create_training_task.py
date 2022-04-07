from sourceRdefs import createTrainingTask, mergeUpdateAnnotations

# merge annotations to update them given changes to other trainset tasks before creating new training tasks
mergeUpdateAnnotations()

# create a new training task, with images staying exclusive from one another until mergeAnnotations is called again
createTrainingTask(trainer="Rob",task_name="4-5-22_Classes1000",n=1000)