from sourceRdefs import createTrainingTask, mergeUpdateAnnotations

# merge annotations creates a new annotations.csv file using completed training tasks (NOTE - this removes the in_task column)
# mergeUpdateAnnotations(proj_root="G:/ant-patterner")
#mergeUpdateAnnotations()

# create a new training task by making a folder and moving images, then adds an in_task column to annotations.csv
# images stay exclusive from one another due to the in_task column (unless mergeAnnotations is called again AND tasks aren't complete
#createTrainingTask(trainer="Jacob",task_name="50_Segments", n=50,name_contains="-h",random_downloaded_only=False,proj_root="G:/ant-patterner")
createTrainingTask(trainer="Ana",task_name="5-23-22_Segments50", n=50,random_downloaded_only=True)