from sourceRdefs import createTrainingTask, mergeUpdateAnnotations

# merge annotations creates a new annotations.csv file using completed training tasks (NOTE - this removes the in_task column)
mergeUpdateAnnotations()

# create a new training task by making a folder and moving images, then adds an in_task column to annotations.csv
# images stay exclusive from one another due to the in_task column (unless mergeAnnotations is called again AND tasks aren't complete)
createTrainingTask(trainer="Shared",task_name="5-10-22_Classes_150", n=150,random_downloaded_only=True)