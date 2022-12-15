from extractHoneSegments import extractHoneSegments
from getFilterImageIDs import getFilterImageIDs
from inferSegments import inferSegments

image_ids = getFilterImageIDs(proj_dir="D:/ant-patterner",contains_str="_d")

inferSegments(image_ids=image_ids,model_name="ant_head_segment",image_size=200,show=False, proj_dir="D:/ant-patterner")

extractHoneSegments(image_ids,bound=True,rotate_to_vertical=False,remove_islands=False,
                    adj_to_background_col=False,
                    set_nonwhite_to_black=True,show=False,write=True,proj_dir="D:/ant-patterner")