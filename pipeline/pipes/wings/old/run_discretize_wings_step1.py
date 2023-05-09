from getFilterImageIDs import getFilterImageIDs
from colorDiscretize import colorDiscretize
from visualizeSegmentPatterns import visualizeSegmentPatterns
import random

ids = getFilterImageIDs(proj_dir="D:/wing-color")
ids = [i + "_hind" for i in ids] + [i + "_fore" for i in ids]
print(ids)
#random.Random(4).shuffle(ids) # 4 is the seed
ids = ids[0:100]

#colorDiscretize(ids,scale=True,use_positions=True,proj_dir="D:/wing-color",write_subfolder="std",vert_resize=80,print_details=False,show=False)

colorDiscretize(ids,preclustered=False, nclusters=3,show=True,bilat_blur=True,gaussian_blur=False,
                median_blur=True, blur_size=5,
                scale=True,proj_dir="D:/wing-color",write_subfolder="test",vert_resize=80,print_details=False)
