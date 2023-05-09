from getFilterImageIDs import getFilterImageIDs
from colorDiscretize import colorDiscretize
from visualizeSegmentPatterns import visualizeSegmentPatterns
import random

ids = getFilterImageIDs(proj_dir="D:/wing-color")
ids = [i + "_fore" for i in ids]
random.Random(1).shuffle(ids) # 4 is the seed
ids = ids[0:10]

# use rgb
colorDiscretize(ids,scale=False,use_positions=False,colorspace="rgb",
                bilat_blur=False, gaussian_blur=True, blur_size=25, blur_sigma=0,
                group_cluster_raw_ids=False,
                write_subfolder="test_rgb", proj_dir="D:/wing-color",vert_resize=100)
# use hls
colorDiscretize(ids,scale=False,use_positions=False,colorspace="hls",
                upweight_axis=2,
                bilat_blur=False, gaussian_blur=True, blur_size=25, blur_sigma=0,
                group_cluster_raw_ids=False,
                write_subfolder="test_hls", proj_dir="D:/wing-color",vert_resize=100)

# debatable
visualizeSegmentPatterns(ids,pattern_dirs=["test_rgb","test_hls"], proj_dir="D:/wing-color")