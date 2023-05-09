from getFilterImageIDs import getFilterImageIDs
from colorDiscretize import colorDiscretize
from visualizeSegmentPatterns import visualizeSegmentPatterns
import random

ids = getFilterImageIDs(proj_dir="D:/wing-color")
ids = [i + "_hind" for i in ids] + [i + "_fore" for i in ids]

ids = random.sample(ids, 100)
colorDiscretize(ids,preclustered=True, group_cluster_raw_ids=True,nclusters=6,
                scale=False,proj_dir="D:/wing-color",bilat_blur=False,write_subfolder="grouped_for_vis",vert_resize=80,print_details=False,show=True)

visualizeSegmentPatterns(ids,proj_dir="D:/wing-color")