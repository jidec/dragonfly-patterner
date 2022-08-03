from getFilterImageIDs import getFilterImageIDs
from inferSegments import inferSegments
from getIDsFromDir import getIDsFromDir

#image_ids = getFilterImageIDs(proj_dir="E:/ant-patterner",contains_str="_h", not_in_inference_data=True)
image_ids = getIDsFromDir("E:/ant-patterner/data/new_images/antweb-images/antweb-images",contains="_h")
inferSegments(image_ids=image_ids,model_name="ant_head_segment",image_size=200,show=True, proj_dir="E:/ant-patterner")

#image_ids = getFilterImageIDs(proj_dir="E:/ant-patterner",contains_str="_d")
#inferSegments(image_ids=image_ids,model_name="ant_gaster_segment",part_suffix="abdomen",image_size=200,show=True, proj_dir="E:/ant-patterner")