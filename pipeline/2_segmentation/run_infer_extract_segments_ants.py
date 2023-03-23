from extractHoneSegments import extractHoneSegments
from getFilterImageIDs import getFilterImageIDs
from inferSegments import inferSegments

image_ids = getFilterImageIDs(proj_dir="D:/ant-patterner",contains_str="_d") #,infer_fields=["bad_signifier"],infer_values=[0])

#image_ids = image_ids[49500:len(image_ids)]

#inferSegments(image_ids=image_ids,model_name="ant_thorax_segment",image_size=200,show=False, proj_dir="D:/ant-patterner")

extractHoneSegments(image_ids,bound=True,rotate_to_vertical=False,remove_islands=False,
                    seg_subfolder="thorax/",
                    set_nonwhite_to_black=True,show=False,print_steps=False,write=True,proj_dir="D:/ant-patterner")

#extractHoneSegments(image_ids,bound=True,rotate_to_vertical=False,remove_islands=False,
#                    adj_to_background_col=True,target_bg_col=[180,180,180],
#                    seg_subfolder="grey_adj/",
#                    set_nonwhite_to_black=True,show=False,target_range=80,print_steps=False,write=True,proj_dir="D:/ant-patterner")

#inferSegments(image_ids=image_ids,model_name="ant_head_segment2",image_size=200,show=False, proj_dir="D:/ant-patterner")

#extractHoneSegments(image_ids,bound=True,rotate_to_vertical=False,remove_islands=False,
#                    adj_to_background_col=False,
#                    set_nonwhite_to_black=True,show=False,write=True,proj_dir="D:/ant-patterner")