from getFilterImageIDs import getFilterImageIDs
from colorDiscretize import colorDiscretize
from visualizeSegmentPatterns import visualizeSegmentPatterns
import random

ids = getFilterImageIDs(proj_dir="D:/wing-color")
ids = [i + "_fore" for i in ids]
random.Random(1).shuffle(ids) # 4 is the seed
ids = ids[0:10]

# default with blur
colorDiscretize(ids,scale=False,use_positions=False,colorspace="rgb",
                bilat_blur=False, gaussian_blur=True, blur_size=15, blur_sigma=0,
                group_cluster_raw_ids=False,
                write_subfolder="test_blur", proj_dir="D:/wing-color",vert_resize=80)
# default no blur
colorDiscretize(ids,scale=False,use_positions=False,colorspace="rgb",
                bilat_blur=False, gaussian_blur=False, blur_size=15, blur_sigma=0,
                group_cluster_raw_ids=False,
                write_subfolder="test_noblur", proj_dir="D:/wing-color",vert_resize=80)

#blur helps
visualizeSegmentPatterns(ids,pattern_dirs=["test_blur"], proj_dir="D:/wing-color")
visualizeSegmentPatterns(ids,pattern_dirs=["test_noblur"], proj_dir="D:/wing-color")