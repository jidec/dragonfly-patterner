from getFilterImageIDs import getFilterImageIDs
from colorDiscretize import colorDiscretize
from visualizeSegmentPatterns import visualizeSegmentPatterns
import random

ids = getFilterImageIDs(proj_dir="D:/wing-color")
ids = [i + "_fore" for i in ids]
random.Random(1).shuffle(ids) # 4 is the seed
ids = ids[0:10]

# default with scale
colorDiscretize(ids,scale=True,use_positions=False,colorspace="rgb",
                bilat_blur=False, gaussian_blur=True, blur_size=25, blur_sigma=0,
                group_cluster_raw_ids=False,
                write_subfolder="test_scale", proj_dir="D:/wing-color",vert_resize=80)
# default no scale
colorDiscretize(ids,scale=False,use_positions=False,colorspace="rgb",
                bilat_blur=False, gaussian_blur=True, blur_size=25, blur_sigma=0,
                group_cluster_raw_ids=False,
                write_subfolder="test_noscale", proj_dir="D:/wing-color",vert_resize=80)

# scaling doesnt help
visualizeSegmentPatterns(ids,pattern_dirs=["test_scale","test_noscale"], proj_dir="D:/wing-color")