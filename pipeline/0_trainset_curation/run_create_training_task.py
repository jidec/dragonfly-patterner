#from sourceRdefs import createTrainingTask, mergeUpdateAnnotations
from getFilterImageIDs import getFilterImageIDs
from createTrainingTask import createTrainingTask
# merge annotations creates a new annotations.csv file using completed training tasks (NOTE - this removes the in_task column)
# mergeUpdateAnnotations(proj_root="G:/ant-patterner")
#mergeUpdateAnnotations()

# create a new training task by making a folder and moving images, then adds an in_task column to annotations.csv
# images stay exclusive from one another due to the in_task column (unless mergeAnnotations is called again AND tasks aren't complete
training_id_pool = getFilterImageIDs(not_in_train_data=True)
createTrainingTask(trainer_name="Jacob",task_name="test",image_id_pool=training_id_pool,num_images=10)

#createTrainingTask(trainer="Jacob",task_name="50_SegmentsAbdomenDorsal", n=50,name_contains="-d",random_downloaded_only=False,proj_root="E:/ant-patterner")
#createTrainingTask(trainer="Ana",task_name="5-23-22_Segments50", n=50,random_downloaded_only=True)