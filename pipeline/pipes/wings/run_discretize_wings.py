from getFilterImageIDs import getFilterImageIDs
from colorDiscretize import colorDiscretize
from visualizeSegmentPatterns import visualizeSegmentPatterns
import random

ids = getFilterImageIDs(proj_dir="D:/wing-color")
ids = [i + "_hind" for i in ids] + [i + "_fore" for i in ids]
#random.Random(4).shuffle(ids) # 4 is the seed
#ids = ids[0:100]

#colorDiscretize(ids,scale=True,use_positions=True,proj_dir="D:/wing-color",write_subfolder="std",vert_resize=80,print_details=False,show=False)
#visualizeSegmentPatterns(ids, pattern_dirs=["grouped2"], proj_dir="D:/wing-color")
colorDiscretize(ids,preclustered=True, group_cluster_raw_ids=True, show=True, show_indv=False,
                nclusters=3, nclust_metric="ch", scale=False, #upweight_axis=0,
                cluster_model="gaussian_mixture",
                #cluster_model="dbscan", cluster_eps=4.5, cluster_min_samples=5, #100n
                #cluster_model="dbscan", cluster_eps=5, cluster_min_samples=15, #500n
                bilat_blur=False,gaussian_blur=False,median_blur=False, colorspace="rgb",
                proj_dir="D:/wing-color",#preclust_read_subfolder="old_patterns",
                write_subfolder="grouped2",vert_resize=40,print_details=False)

visualizeSegmentPatterns(ids, pattern_dirs=["","grouped2"], proj_dir="D:/wing-color")